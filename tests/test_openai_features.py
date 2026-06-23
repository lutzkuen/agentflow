import asyncio
import copy
import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.optimization import openai_features
from tokenclaw.optimization import openai_pipeline
from tokenclaw import router as router_module
from tokenclaw import cache as cache_module
from tokenclaw.openai_routing_report import build_openai_routing_report
import tokenclaw.routing_experiments as routing_experiments_module
from tokenclaw.store import stable_json


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from tokenclaw import openai_proxy, server
    from tokenclaw.store import Store


class FakeJsonResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = stable_json(body).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self):
        return self._body


class CapturingOpenAIClient:
    calls = []
    response_body = {}
    status_code = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        self.__class__.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        return FakeJsonResponse(self.__class__.response_body, self.__class__.status_code)


class FakeStreamResponse:
    def __init__(self, chunks, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class CapturingStreamingOpenAIClient(CapturingOpenAIClient):
    stream_calls = []
    stream_chunks = [
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":11,"output_tokens":3}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    stream_status_code = 200

    def stream(self, method, url, *, headers=None, json=None, **kwargs):
        self.__class__.stream_calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers or {}),
            "json": copy.deepcopy(json),
            "kwargs": kwargs,
        })
        return FakeStreamResponse(
            self.__class__.stream_chunks,
            status_code=self.__class__.stream_status_code,
        )


class RateLimitThenSuccessOpenAIClient(CapturingOpenAIClient):
    calls = []

    async def post(self, url, *, headers=None, json=None, **kwargs):
        self.__class__.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": copy.deepcopy(json),
            "kwargs": kwargs,
        })
        if len(self.__class__.calls) == 1:
            return FakeJsonResponse(
                {"error": {"message": "rate limited"}},
                status_code=429,
                headers={"content-type": "application/json", "retry-after": "0"},
            )
        return FakeJsonResponse(self.__class__.response_body, self.__class__.status_code)


class OpenAIFeatureUnitTests(unittest.TestCase):
    def _assert_no_raw_values(self, value):
        text = str(value)
        for raw in (
            "raw openai prompt",
            "raw chat message",
            "raw function args",
            "raw tool output",
            "secret session id",
            "tenant-secret",
            "api-key-secret",
        ):
            self.assertNotIn(raw, text)

    def test_responses_request_unit_normalizes_metadata_without_raw_payloads(self):
        unit = openai_features.build_openai_request_feature_unit(
            body={
                "model": "gpt-5-codex",
                "input": [
                    {"type": "message", "content": "raw openai prompt"},
                    {"type": "function_call", "arguments": "raw function args"},
                ],
                "tools": [{"type": "function", "name": "lookup", "description": "raw tool output"}],
                "stream": True,
            },
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-codex",
            routing_meta={
                "text_chars": 2048,
                "has_tools": False,
                "category": "chat",
                "policy_source": "local-default",
                "tenant_id": "tenant-secret",
            },
            crunch_meta={"changed": False},
            cache_meta={"status": "skipped", "reason": "streaming"},
            category="chat",
            stream=True,
            input_tokens_est=512,
            session_id="secret session id",
        )
        summary = openai_features.summarize_openai_request_feature_unit(unit)

        self.assertEqual(unit["source_surface"], "openai_responses")
        self.assertEqual(unit["endpoint"], "responses")
        self.assertEqual(unit["provider"], "openai")
        self.assertEqual(unit["requested_model_family"], "gpt-5-codex")
        self.assertEqual(unit["routed_model_family"], "gpt-5-codex")
        self.assertEqual(unit["app_family"], "codex")
        self.assertTrue(unit["tool_features"]["has_tools"])
        self.assertEqual(unit["tool_features"]["declared_tool_count"], 1)
        self.assertEqual(unit["tool_features"]["response_tool_item_types"], ["function_call"])
        self.assertTrue(unit["privacy_summary"]["metadata_only"])
        self.assertFalse(unit["privacy_summary"]["raw_body_storage"])
        self.assertTrue(summary["has_tools"])
        self.assertEqual(summary["source_surface"], "openai_responses")
        self.assertEqual(summary["endpoint"], "responses")
        self.assertFalse(summary["raw_payload_included"])
        self.assertEqual(managed_egress_violations(unit), [])
        self._assert_no_raw_values(unit)
        self._assert_no_raw_values(summary)

    def test_chat_request_unit_detects_chat_tool_calls_without_raw_payloads(self):
        unit = openai_features.build_openai_request_feature_unit(
            body={
                "model": "gpt-4.1",
                "messages": [
                    {"role": "user", "content": "raw chat message"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "raw function args"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "raw tool output"},
                ],
            },
            path="/v1/chat/completions",
            requested_model="gpt-4.1",
            routed_model="gpt-4.1-mini",
            routing_meta={"text_chars": 800, "has_tools": False, "category": "chat"},
            crunch_meta={"changed": False},
            cache_meta={"status": "miss", "reason": "exact-miss"},
            category="chat",
            stream=False,
            input_tokens_est=200,
            session_id=None,
        )
        summary = openai_features.summarize_openai_request_feature_unit(unit)

        self.assertEqual(unit["source_surface"], "openai_chat")
        self.assertEqual(unit["endpoint"], "chat_completions")
        self.assertEqual(unit["requested_model_family"], "gpt-4")
        self.assertEqual(unit["routed_model_family"], "gpt-4")
        self.assertTrue(unit["tool_features"]["has_tools"])
        self.assertEqual(unit["tool_features"]["chat_tool_call_count"], 1)
        self.assertEqual(unit["tool_features"]["chat_tool_result_count"], 1)
        self.assertEqual(summary["chat_tool_call_count"], 1)
        self.assertEqual(summary["chat_tool_result_count"], 1)
        self.assertEqual(managed_egress_violations(unit), [])
        self._assert_no_raw_values(unit)
        self._assert_no_raw_values(summary)

    def test_preflight_unit_is_feature_only_before_local_mutation(self):
        unit = openai_features.build_openai_preflight_feature_unit(
            body={
                "model": "gpt-5-codex",
                "input": [
                    {"role": "system", "content": "raw openai prompt"},
                    {
                        "type": "message",
                        "content": "\n".join([
                            "raw chat message /home/lutz/project/app.py",
                            "2026-06-09T20:00:00Z INFO pid=1234 secret-app started",
                            "2026-06-09T20:00:01Z ERROR pid=1234 secret-app failed",
                        ]),
                    },
                    {"type": "function_call", "arguments": "raw function args"},
                ],
                "tools": [{"type": "function", "name": "lookup", "description": "raw tool output"}],
                "metadata": {
                    "request_id": "req_raw_secret",
                    "session_id": "secret session id",
                    "cache_key": "cache-key-secret",
                    "api_key": "api-key-secret",
                },
                "stream": True,
            },
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routing_meta={
                "text_chars": 4096,
                "has_tools": True,
                "category": "tool-light",
                "workflow_phase": "tool-execution",
            },
            category="tool-light",
            stream=True,
            input_tokens_est=1024,
        )
        summary = openai_features.summarize_openai_request_feature_unit(unit)
        rendered = json.dumps(unit, sort_keys=True)

        self.assertEqual(unit["schema"], "tokenclaw.openai_preflight_feature_unit.v1")
        self.assertEqual(unit["source_surface"], "openai_responses")
        self.assertEqual(unit["candidate_target_model"], None)
        self.assertEqual(unit["grouping_identifiers"], {})
        self.assertEqual(unit["input_features"]["local_mutation_stage"], "preflight")
        self.assertEqual(unit["input_features"]["path_class"], "responses")
        self.assertEqual(unit["input_features"]["old_context"]["shape"], "responses_input_items")
        self.assertTrue(unit["input_features"]["cache_eligibility"]["streaming_bypass_hint"])
        self.assertFalse(unit["input_features"]["cache_eligibility"]["raw_cache_key_included"])
        terminal_features = unit["input_features"]["terminal_log_features"]
        difficulty_features = unit["input_features"]["prompt_difficulty_features"]
        self.assertEqual(terminal_features["schema"], "tokenclaw.terminal_log_features.v1")
        self.assertEqual(terminal_features["log_line_fraction_bucket"], "25_50pct")
        self.assertEqual(terminal_features["error_line_count_bucket"], "one")
        self.assertFalse(terminal_features["privacy"]["raw_log_text_included"])
        self.assertEqual(difficulty_features["schema"], "tokenclaw.prompt_difficulty_features.v1")
        self.assertEqual(difficulty_features["downgrade_risk"], "block")
        self.assertEqual(difficulty_features["external_source_dependency"], "logs")
        self.assertEqual(summary["local_mutation_stage"], "preflight")
        self.assertEqual(summary["terminal_log_features"], terminal_features)
        self.assertEqual(summary["prompt_difficulty_features"], difficulty_features)
        self.assertTrue(summary["has_tools"])
        for forbidden_key in ("messages", "input", "cache_key", "request_id", "session_id"):
            self.assertNotIn(f'"{forbidden_key}"', rendered)
        self.assertEqual(managed_egress_violations(unit), [])
        self._assert_no_raw_values(unit)
        self.assertNotIn("/home/lutz/project/app.py", rendered)
        self.assertNotIn("secret-app", rendered)

    def test_pipeline_preflight_stage_is_guarded_feature_only_metadata(self):
        body = {
            "model": "gpt-5-codex",
            "input": [
                {"type": "message", "content": "raw openai prompt"},
                {"type": "function_call", "arguments": "raw function args"},
            ],
            "tools": [{"type": "function", "name": "lookup", "description": "raw tool output"}],
            "metadata": {"session_id": "secret session id", "api_key": "api-key-secret"},
            "stream": True,
        }

        parsed = openai_pipeline.parse_openai_request_body(body, ["gpt-5-codex"])
        preflight = openai_pipeline.extract_openai_preflight_features(parsed, path="/v1/responses")

        self.assertEqual(parsed.stage, "parse_request")
        self.assertEqual(preflight.stage, "extract_preflight_features")
        self.assertEqual(preflight.feature_unit["schema"], "tokenclaw.openai_preflight_feature_unit.v1")
        self.assertEqual(preflight.request_facts["schema"], "tokenclaw.request_facts_envelope.v1")
        self.assertEqual(preflight.feature_summary["local_mutation_stage"], "preflight")
        self.assertEqual(managed_egress_violations(preflight.feature_unit), [])
        self.assertEqual(managed_egress_violations(preflight.request_facts), [])
        self.assertEqual(managed_egress_violations(preflight.feature_summary), [])
        self._assert_no_raw_values(preflight.feature_unit)
        self._assert_no_raw_values(preflight.request_facts)
        self._assert_no_raw_values(preflight.feature_summary)

    def test_pipeline_policy_fetch_uses_only_guarded_preflight_unit(self):
        parsed = openai_pipeline.parse_openai_request_body(
            {"model": "gpt-5-codex", "input": "raw openai prompt"},
            ["gpt-5-codex"],
        )
        preflight = openai_pipeline.extract_openai_preflight_features(parsed, path="/v1/responses")
        seen = {}

        async def fake_fetcher(*, recommendation_unit, request_facts, current_model, input_tokens_est):
            seen["unit"] = recommendation_unit
            seen["request_facts"] = request_facts
            seen["current_model"] = current_model
            seen["input_tokens_est"] = input_tokens_est
            return {"status": "skipped", "apply_reason": "test-fallback"}

        decision = asyncio.run(openai_pipeline.fetch_openai_policy_decision(preflight, fetcher=fake_fetcher))

        self.assertEqual(decision["apply_reason"], "test-fallback")
        self.assertIs(seen["unit"], preflight.feature_unit)
        self.assertIs(seen["request_facts"], preflight.request_facts)
        self.assertEqual(seen["current_model"], "gpt-5-codex")
        self.assertEqual(managed_egress_violations(seen["unit"]), [])
        self.assertEqual(managed_egress_violations(seen["request_facts"]), [])
        self._assert_no_raw_values(seen["unit"])
        self._assert_no_raw_values(seen["request_facts"])

    def test_pipeline_local_policy_stage_applies_local_actions_without_provider_io(self):
        body = {
            "model": "gpt-5-codex",
            "input": "raw openai prompt\n2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
        }
        parsed = openai_pipeline.parse_openai_request_body(body, ["gpt-5-codex"])
        preflight = openai_pipeline.extract_openai_preflight_features(parsed, path="/v1/responses")
        events = []

        def fake_crunch(raw_body, *, store_obj=None):
            events.append("crunch")
            crunched = copy.deepcopy(raw_body)
            crunched["input"] = "locally crunched prompt"
            return crunched, {"changed": True, "saved_chars": 5, "status": "applied"}

        def fake_router(provider_body):
            events.append("route")
            return "gpt-5-codex", {
                "enabled": True,
                "requested_model": provider_body["model"],
                "routed_model": "gpt-5-codex",
                "reason": "test local route",
                "text_chars": 23,
                "has_tools": False,
                "category": "chat",
                "policy_source": "local-default",
            }

        decision = {
            "schema": "tokenclaw.openai_managed_recommendation_decision.v1",
            "status": "selected",
            "apply_reason": "live-route-selected",
            "selected_for_local_application": True,
            "live_promotion_mode": "live",
            "target_model_normalized": "gpt-5-mini",
            "policy_id": "test-policy",
            "reason": "test policy",
            "local_actions": {"status": "applied"},
        }
        local = openai_pipeline.execute_openai_local_policy(
            raw_body=body,
            path="/v1/responses",
            requested_model=parsed.requested_model,
            category=parsed.category,
            stream=parsed.stream,
            session_id="secret session id",
            preflight=preflight,
            policy_decision=decision,
            store_obj=None,
            cruncher=fake_crunch,
            router=fake_router,
        )

        self.assertEqual(events, ["crunch", "route"])
        self.assertEqual(local.stage, "execute_local_policy")
        self.assertEqual(local.provider_body["model"], "gpt-5-mini")
        self.assertEqual(local.provider_body["input"], "locally crunched prompt")
        self.assertEqual(local.routing_meta["managed_recommendation"]["status"], "applied")
        self.assertEqual(local.routing_meta["openai_feature_unit"]["local_mutation_stage"], "preflight")
        self.assertEqual(
            local.routing_meta["openai_feature_unit"]["terminal_log_features"]["error_line_count_bucket"],
            "one",
        )
        self.assertEqual(
            local.routing_meta["openai_feature_unit"]["prompt_difficulty_features"]["downgrade_risk"],
            "block",
        )
        self.assertEqual(local.routing_meta["openai_local_feature_unit"]["source_surface"], "openai_responses")
        self.assertNotIn("raw openai prompt", json.dumps(local.routing_meta, sort_keys=True))
        self.assertNotIn("secret-app", json.dumps(local.routing_meta, sort_keys=True))

    def test_pipeline_local_policy_stage_records_post_local_pattern_diagnostics(self):
        raw_secret = "raw openai prompt secret"
        body = {
            "model": "gpt-5-codex",
            "input": (
                f"{raw_secret}\n"
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-old\n"
                "+new\n"
            ),
        }
        parsed = openai_pipeline.parse_openai_request_body(body, ["gpt-5-codex"])
        preflight = openai_pipeline.extract_openai_preflight_features(parsed, path="/v1/responses")

        local = openai_pipeline.execute_openai_local_policy(
            raw_body=body,
            path="/v1/responses",
            requested_model=parsed.requested_model,
            category=parsed.category,
            stream=parsed.stream,
            session_id="secret session id",
            preflight=preflight,
            policy_decision={"status": "skipped", "reason": "disabled"},
            store_obj=None,
        )

        diagnostics = local.routing_meta["managed_pattern_features"]
        self.assertTrue(diagnostics["present"])
        self.assertEqual(diagnostics["source_surface"], "openai_responses")
        self.assertIn("diffs", diagnostics["local_pattern_module_families"])
        self.assertTrue(diagnostics["pattern_hash"].startswith("sha256:"))
        self.assertFalse(diagnostics["raw_pattern_strings_included"])
        self.assertEqual(managed_egress_violations(diagnostics), [])
        rendered = json.dumps(local.routing_meta, sort_keys=True)
        self.assertNotIn(raw_secret, rendered)
        self.assertNotIn("secret session id", rendered)

    def test_pipeline_outcome_serialization_is_feature_only_and_guarded(self):
        summary = openai_pipeline.serialize_openai_outcome_summary(
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            status_code=200,
            latency_ms=10,
            retry_count=0,
            input_tokens_est=20,
            output_tokens_est=3,
            actual_input_tokens=21,
            actual_output_tokens=4,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            cache_meta={"status": "miss", "reason": "exact-miss"},
            crunch_meta={"changed": False},
            routing_meta={
                "reason": "test",
                "managed_recommendation": {"status": "applied"},
                "openai_feature_unit": {
                    "terminal_log_features": {
                        "schema": "tokenclaw.terminal_log_features.v1",
                        "terminal_output_char_fraction_bucket": "gte_75pct",
                        "raw_log_text_included": False,
                    },
                    "prompt_difficulty_features": {
                        "schema": "tokenclaw.prompt_difficulty_features.v1",
                        "task_intent": "data_lookup",
                        "downgrade_risk": "block",
                        "privacy": {"metadata_only": True},
                    },
                },
            },
            category="chat",
            session_id="secret session id",
            error=None,
        )

        self.assertEqual(summary["schema"], "tokenclaw.openai_outcome_summary.v1")
        self.assertEqual(summary["status_code"], 200)
        self.assertEqual(summary["terminal_log_features"]["terminal_output_char_fraction_bucket"], "gte_75pct")
        self.assertEqual(summary["prompt_difficulty_features"]["downgrade_risk"], "block")
        self.assertFalse(summary["raw_payload_included"])
        self.assertEqual(managed_egress_violations(summary), [])
        self._assert_no_raw_values(summary)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class OpenAIFeatureRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.cwd_tmp = tempfile.TemporaryDirectory()
        os.chdir(self.cwd_tmp.name)
        self.old_store = server.store
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.old_log_bodies = server.LOG_BODIES
        self.saved_recommendation_enabled = os.environ.get("TOKENCLAW_RECOMMENDATION_ENABLED")
        self.saved_openai_recommendation_mode = os.environ.get("TOKENCLAW_OPENAI_RECOMMENDATION_MODE")
        self.saved_openai_recommendation_canary_fraction = os.environ.get("TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION")
        self.saved_policy_decision_enabled = os.environ.get("TOKENCLAW_POLICY_DECISION_ENABLED")
        self.saved_policy_decision_canary_fraction = os.environ.get("TOKENCLAW_POLICY_DECISION_CANARY_FRACTION")
        self.saved_routing_rules = os.environ.get("TOKENCLAW_ROUTING_RULES")
        self.saved_crunch_rules = os.environ.get("TOKENCLAW_CRUNCH_RULES")
        self.saved_cache_env = {
            key: os.environ.get(key)
            for key in (
                "TOKENCLAW_CACHE",
                "TOKENCLAW_CACHE_TOOL_CALLS",
                "TOKENCLAW_SEMANTIC_CACHE",
                "TOKENCLAW_SEMANTIC_THRESHOLD",
                "TOKENCLAW_CACHE_RULES",
                "TOKENCLAW_CACHE_CANARY_POLICY",
                "TOKENCLAW_CACHE_NAMESPACE",
                "TOKENCLAW_CACHE_FILE_WATCH",
                "TOKENCLAW_CACHE_WATCH_ROOT",
                "TOKENCLAW_CACHE_WATCH_MAX_PATHS",
                "TOKENCLAW_CACHE_CAPTURE_CANDIDATES",
                "TOKENCLAW_CACHE_TTL_SECONDS",
                "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP",
                "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP_WINDOW",
            )
        }
        self.saved_routing_experiments = os.environ.get("TOKENCLAW_ROUTING_EXPERIMENTS")
        self.saved_openai_summary_enabled = os.environ.get("TOKENCLAW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED")
        self.saved_tokenclaw_db = os.environ.get("TOKENCLAW_DB")
        os.environ.pop("TOKENCLAW_RECOMMENDATION_ENABLED", None)
        os.environ.pop("TOKENCLAW_OPENAI_RECOMMENDATION_MODE", None)
        os.environ.pop("TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION", None)
        os.environ.pop("TOKENCLAW_POLICY_DECISION_ENABLED", None)
        os.environ.pop("TOKENCLAW_POLICY_DECISION_CANARY_FRACTION", None)
        os.environ.pop("TOKENCLAW_ROUTING_RULES", None)
        os.environ.pop("TOKENCLAW_ROUTING_EXPERIMENTS", None)
        os.environ.pop("TOKENCLAW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED", None)
        for key in self.saved_cache_env:
            os.environ.pop(key, None)
        importlib.reload(cache_module)
        importlib.reload(routing_experiments_module)
        importlib.reload(router_module)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)
        server.LOG_BODIES = False
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        CapturingOpenAIClient.calls = []
        CapturingOpenAIClient.status_code = 200
        CapturingOpenAIClient.response_body = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

    def tearDown(self):
        if self.saved_recommendation_enabled is None:
            os.environ.pop("TOKENCLAW_RECOMMENDATION_ENABLED", None)
        else:
            os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = self.saved_recommendation_enabled
        if self.saved_openai_recommendation_mode is None:
            os.environ.pop("TOKENCLAW_OPENAI_RECOMMENDATION_MODE", None)
        else:
            os.environ["TOKENCLAW_OPENAI_RECOMMENDATION_MODE"] = self.saved_openai_recommendation_mode
        if self.saved_openai_recommendation_canary_fraction is None:
            os.environ.pop("TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION", None)
        else:
            os.environ["TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION"] = self.saved_openai_recommendation_canary_fraction
        if self.saved_policy_decision_enabled is None:
            os.environ.pop("TOKENCLAW_POLICY_DECISION_ENABLED", None)
        else:
            os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = self.saved_policy_decision_enabled
        if self.saved_policy_decision_canary_fraction is None:
            os.environ.pop("TOKENCLAW_POLICY_DECISION_CANARY_FRACTION", None)
        else:
            os.environ["TOKENCLAW_POLICY_DECISION_CANARY_FRACTION"] = self.saved_policy_decision_canary_fraction
        if self.saved_routing_rules is None:
            os.environ.pop("TOKENCLAW_ROUTING_RULES", None)
        else:
            os.environ["TOKENCLAW_ROUTING_RULES"] = self.saved_routing_rules
        if self.saved_crunch_rules is None:
            os.environ.pop("TOKENCLAW_CRUNCH_RULES", None)
        else:
            os.environ["TOKENCLAW_CRUNCH_RULES"] = self.saved_crunch_rules
        for key, value in self.saved_cache_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self.saved_routing_experiments is None:
            os.environ.pop("TOKENCLAW_ROUTING_EXPERIMENTS", None)
        else:
            os.environ["TOKENCLAW_ROUTING_EXPERIMENTS"] = self.saved_routing_experiments
        if self.saved_openai_summary_enabled is None:
            os.environ.pop("TOKENCLAW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED", None)
        else:
            os.environ["TOKENCLAW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED"] = self.saved_openai_summary_enabled
        if self.saved_tokenclaw_db is None:
            os.environ.pop("TOKENCLAW_DB", None)
        else:
            os.environ["TOKENCLAW_DB"] = self.saved_tokenclaw_db
        importlib.reload(router_module)
        importlib.reload(cache_module)
        importlib.reload(routing_experiments_module)
        os.chdir(self.old_cwd)
        self.cwd_tmp.cleanup()
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server.LOG_BODIES = self.old_log_bodies
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_anthropic_upstream,
            openai_upstream=self.old_openai_upstream,
            openai_auth_mode=self.old_openai_auth_mode,
        )

    def _enable_openai_canary(
        self,
        *,
        canary_fraction=1.0,
        holdout_fraction=0.0,
        target_model="gpt-5-mini",
        model_pattern="gpt-5",
        allow_stream=False,
        safety_stop_enabled=False,
        eligible_categories=None,
        cohort_unit="request_features_sequence",
    ):
        policy_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        categories = eligible_categories or ["chat", "short-completion"]
        policy_file.write("\n".join([
            "openai_canary:",
            "  enabled: true",
            "  policy_id: test-openai-local-canary",
            "  promotion_action_id: test-openai-action",
            "  target_candidate_id: test-openai-candidate",
            f"  model_pattern: {model_pattern}",
            f"  target_model: {target_model}",
            "  eligible_categories:",
            *[f"    - {category}" for category in categories],
            "  excluded_categories: []",
            "  allow_tools: false",
            f"  allow_stream: {'true' if allow_stream else 'false'}",
            "  min_text_chars: 0",
            "  max_text_chars: 8000",
            "  min_input_tokens_est: 0",
            "  max_input_tokens_est: 2000",
            f"  canary_fraction: {canary_fraction}",
            f"  holdout_fraction: {holdout_fraction}",
            "  salt: test-openai-local-canary-salt",
            f"  cohort_unit: {cohort_unit}",
            "  safety_stop:",
            f"    enabled: {'true' if safety_stop_enabled else 'false'}",
            "rules: []",
            "",
        ]))
        policy_file.close()
        os.environ["TOKENCLAW_ROUTING_RULES"] = policy_file.name
        importlib.reload(router_module)
        self.addCleanup(lambda: os.path.exists(policy_file.name) and os.unlink(policy_file.name))
        return policy_file.name

    def _openai_cache_pattern_hash(self, request_body, path="/v1/responses"):
        body = copy.deepcopy(request_body)
        parsed = openai_pipeline.parse_openai_request_body(body, list(server.OPENAI_MODEL_LIST))
        preflight = openai_pipeline.extract_openai_preflight_features(parsed, path=path)
        local = openai_pipeline.execute_openai_local_policy(
            raw_body=body,
            path=path,
            requested_model=parsed.requested_model,
            category=parsed.category,
            stream=parsed.stream,
            session_id="test-session",
            preflight=preflight,
            policy_decision={},
            store_obj=server.store,
        )
        return local.routing_meta["managed_pattern_features"]["cache_pattern_hash"]

    def test_openai_pattern_features_include_hashed_rollout_unit_for_canary_split(self):
        first = {
            "model": "gpt-5.4-mini",
            "input": "Summarize the stable release checklist for cache replay.",
        }
        second = {
            "model": "gpt-5.4-mini",
            "input": "Summarize the stable release checklist for cache replay with a different suffix.",
        }

        def diagnostics_for(body):
            parsed = openai_pipeline.parse_openai_request_body(copy.deepcopy(body), list(server.OPENAI_MODEL_LIST))
            preflight = openai_pipeline.extract_openai_preflight_features(parsed, path="/v1/responses")
            local = openai_pipeline.execute_openai_local_policy(
                raw_body=copy.deepcopy(body),
                path="/v1/responses",
                requested_model=parsed.requested_model,
                category=parsed.category,
                stream=parsed.stream,
                session_id="test-session",
                preflight=preflight,
                policy_decision={},
                store_obj=server.store,
            )
            return local.routing_meta["managed_pattern_features"]

        first_features = diagnostics_for(first)
        second_features = diagnostics_for(second)

        self.assertTrue(first_features["rollout_unit_hash"].startswith("sha256:"))
        self.assertTrue(first_features["rollout_unit_hash_included"])
        self.assertFalse(first_features["raw_request_fingerprint_included"])
        self.assertNotEqual(first_features["rollout_unit_hash"], second_features["rollout_unit_hash"])
        rendered = json.dumps({"first": first_features, "second": second_features}, sort_keys=True)
        self.assertNotIn(first["input"], rendered)
        self.assertNotIn(second["input"], rendered)

    def _enable_openai_cache_replay_rule(
        self,
        pattern_hash,
        *,
        source_surface="openai_responses",
        endpoint="responses",
        category="tool-light",
        has_tools=True,
        canary_fraction=1.0,
        graduation=None,
        replayability_levels=None,
    ):
        rule = {
                "id": "reviewed-openai-cache-replay",
                "enabled": True,
                "policy_source": "managed-recommended",
                "candidate_id": "openai-cache-replay-candidate",
                "conditions": {
                    "pattern_hashes": [pattern_hash],
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "app_family": "codex",
                    "category": category,
                    "has_tools": has_tools,
                    "stream": False,
                    "replayability_levels": replayability_levels or ["local-exact-response"],
                },
                "rollout": {
                    "schema": "tokenclaw.pattern_policy_rollout.v1",
                    "recommendation_mode": "canary-only",
                    "canary_enabled": True,
                    "canary_fraction": canary_fraction,
                    "canary_salt": "openai-cache-replay-test",
                    "canary_unit": "request_fingerprint",
                },
                "action": {
                    "type": "exact_cache_pattern",
                    "allow_tool_calls": has_tools,
                    "safe_invalidation": has_tools,
                    "scope": "session",
                },
            }
        if graduation is not None:
            rule["graduation"] = graduation
        rules = cache_module.normalize_cache_pattern_rules([rule])
        old_rules = cache_module.CACHE_PATTERN_RULES
        cache_module.CACHE_PATTERN_RULES = tuple(rules)
        self.addCleanup(lambda: setattr(cache_module, "CACHE_PATTERN_RULES", old_rules))
        return rules[0]

    def _enable_staged_openai_cache_replay_shape_rule(self, *, canary_fraction=1.0, holdout_fraction=0.0):
        rule = {
            "id": "request-shape-cache-replay-test-policy",
            "enabled": True,
            "policy_source": "local-manual",
            "candidate_id": "request-shape-cache-replay-test-cohort",
            "conditions": {
                "pattern_hashes": ["sha256:*"],
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "chat",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "has_tools": False,
                "stream": False,
                "replayability_levels": ["features_only", "local-exact-response"],
            },
            "rollout": {
                "schema": "tokenclaw.pattern_policy_rollout.v1",
                "recommendation_mode": "openai-cache-replay-request-shape-canary",
                "canary_enabled": True,
                "canary_fraction": canary_fraction,
                "holdout_fraction": holdout_fraction,
                "canary_salt": "request-shape-cache-replay-test-policy",
                "canary_unit": "request_fingerprint",
            },
            "action": {
                "type": "exact_cache_pattern",
                "allow_tool_calls": False,
                "safe_invalidation": False,
                "streaming": False,
                "scope": "session",
                "min_call_count": 2,
                "ttl_seconds": 3600,
            },
            "graduation": {
                "schema": "tokenclaw.request_shape_cache_replay_shape_activation.v1",
                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                "source_reason": "replay-ready-exact-non-tool-shape",
                "cohort_id": "request-shape-cache-replay-test-cohort",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "chat",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "projected_hits": 35,
                "projected_savings_usd": 0.075373,
                "sample_count": 36,
                "aggregate_only": True,
            },
        }
        rules = cache_module.normalize_cache_pattern_rules([rule])
        old_rules = cache_module.CACHE_PATTERN_RULES
        cache_module.CACHE_PATTERN_RULES = tuple(rules)
        self.addCleanup(lambda: setattr(cache_module, "CACHE_PATTERN_RULES", old_rules))
        return rules[0]

    def test_openai_cache_pattern_rule_preserves_sanitized_graduation_projection(self):
        rules = cache_module.normalize_cache_pattern_rules([
            {
                "id": "reviewed-openai-cache-replay",
                "enabled": True,
                "policy_source": "managed-recommended",
                "candidate_id": "openai-cache-replay-candidate",
                "conditions": {
                    "pattern_hashes": ["sha256:" + "c" * 64],
                    "source_surface": "openai_responses",
                    "endpoint": "responses",
                    "category": "chat",
                    "has_tools": False,
                    "stream": False,
                },
                "rollout": {
                    "schema": "tokenclaw.pattern_policy_rollout.v1",
                    "recommendation_mode": "canary-only",
                    "canary_enabled": True,
                    "canary_fraction": 1.0,
                    "canary_salt": "openai-cache-replay-test",
                    "canary_unit": "request_fingerprint",
                },
                "action": {
                    "type": "exact_cache_pattern",
                    "allow_tool_calls": False,
                    "safe_invalidation": False,
                    "scope": "session",
                },
                "graduation": {
                    "schema": "tokenclaw.openai_cache_replay_shape_activation.v1",
                    "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                    "source_reason": "replay-ready-exact-non-tool-shape",
                    "projected_hits": 3,
                    "projected_savings_usd": 0.09,
                    "sample_count": 4,
                    "aggregate_only": True,
                    "raw_prompt": "raw normalized projection must not leak",
                },
            }
        ])

        self.assertEqual(rules[0]["graduation"]["projected_hits"], 3)
        self.assertEqual(rules[0]["graduation"]["sample_count"], 4)
        self.assertEqual(rules[0]["graduation"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        rendered = json.dumps(rules, sort_keys=True)
        self.assertNotIn("raw normalized projection must not leak", rendered)
        self.assertFalse(rules[0]["graduation"]["raw_prompts_included"])
        self.assertFalse(rules[0]["graduation"]["cache_keys_included"])

    def test_openai_cache_replay_canary_applied_serves_cached_responses_response(self):
        watched = os.path.join(self.cwd_tmp.name, "src", "example.py")
        os.makedirs(os.path.dirname(watched), exist_ok=True)
        with open(watched, "w", encoding="utf-8") as f:
            f.write("print('stable replay dependency')\n")
        request_body = {
            "model": "gpt-5-codex",
            "input": "Inspect ./src/example.py with the lookup tool and summarize the result.",
            "tools": [{"type": "function", "name": "lookup_file"}],
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body)
        self._enable_openai_cache_replay_rule(pattern_hash)
        CapturingOpenAIClient.response_body = {
            "id": "resp_cached",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "cached response"}]}],
            "usage": {"input_tokens": 25, "output_tokens": 5},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )
            CapturingOpenAIClient.response_body = {
                "id": "resp_upstream_should_not_be_used",
                "object": "response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "not used"}]}],
                "usage": {"input_tokens": 25, "output_tokens": 5},
            }
            second = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], "resp_cached")
        self.assertEqual(len(CapturingOpenAIClient.calls), 1)
        rows = server.store.conn.execute("select cache_hit, cache_json from calls order by rowid").fetchall()
        self.assertEqual(rows[0]["cache_hit"], 0)
        self.assertEqual(rows[1]["cache_hit"], 1)
        seed_cache = json.loads(rows[0]["cache_json"])
        hit_cache = json.loads(rows[1]["cache_json"])
        self.assertEqual(seed_cache["cache_replay_store"]["status"], "stored")
        self.assertFalse(seed_cache["cache_replay_store"]["cache_key_included"])
        self.assertEqual(hit_cache["status"], "hit")
        self.assertEqual(hit_cache["cache_replay_canary"]["status"], "applied")
        self.assertEqual(hit_cache["cache_replay_canary"]["canary_cohort"], "canary_applied")
        self.assertEqual(hit_cache["pattern_rule"]["rule_id"], "reviewed-openai-cache-replay")
        self.assertEqual(hit_cache["pattern_rule"]["candidate_id"], "openai-cache-replay-candidate")
        self.assertGreater(hit_cache["estimated_saved_cost_usd"], 0)
        rendered = json.dumps({"seed": seed_cache, "hit": hit_cache}, sort_keys=True)
        self.assertNotIn("src/example.py", rendered)
        self.assertNotIn(watched, rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)

    def test_openai_cache_replay_canary_records_first_real_non_tool_responses_hit(self):
        CapturingOpenAIClient.calls = []
        request_body = {
            "model": "gpt-5-codex",
            "input": " ".join(["Summarize the stable release checklist for this offline replay canary."] * 60),
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body)
        self._enable_openai_cache_replay_rule(
            pattern_hash,
            category="chat",
            has_tools=False,
            replayability_levels=["features_only", "local-exact-response"],
            graduation={
                "schema": "tokenclaw.openai_cache_replay_shape_activation.v1",
                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                "source_reason": "replay-ready-exact-non-tool-shape",
                "cohort_bucket": "openai_responses/responses/chat/chat/2k_8k_chars/500_2k_tokens",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "chat",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "projected_hits": 1,
                "projected_savings_usd": 0.01,
                "sample_count": 2,
                "aggregate_only": True,
            },
        )
        CapturingOpenAIClient.response_body = {
            "id": "resp_non_tool_cached",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "cached non-tool response"}]}],
            "usage": {"input_tokens": 25, "output_tokens": 5},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )
            CapturingOpenAIClient.response_body = {
                "id": "resp_non_tool_upstream_should_not_be_used",
                "object": "response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "not used"}]}],
                "usage": {"input_tokens": 25, "output_tokens": 5},
            }
            second = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], "resp_non_tool_cached")
        rows = server.store.conn.execute("select cache_hit, stream, cache_json from calls order by rowid").fetchall()
        self.assertEqual(sum(1 for row in rows if row["cache_hit"] == 1), 1)
        self.assertEqual(rows[0]["cache_hit"], 0)
        self.assertEqual(rows[1]["cache_hit"], 1)
        self.assertEqual(rows[0]["stream"], 0)
        self.assertEqual(rows[1]["stream"], 0)
        seed_cache = json.loads(rows[0]["cache_json"])
        hit_cache = json.loads(rows[1]["cache_json"])
        self.assertEqual(seed_cache["status"], "miss")
        self.assertEqual(seed_cache["reason"], "exact-pattern-miss")
        self.assertEqual(seed_cache["cache_replay_store"]["status"], "stored")
        self.assertEqual(seed_cache["cache_replay_canary"]["status"], "applied")
        self.assertEqual(hit_cache["status"], "hit")
        self.assertEqual(hit_cache["reason"], "exact-match")
        self.assertEqual(hit_cache["cache_replay_canary"]["status"], "applied")
        self.assertEqual(hit_cache["cache_replay_canary"]["projection"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(hit_cache["pattern_rule"]["allow_tool_calls"], False)
        self.assertEqual(hit_cache["pattern_rule"]["safe_invalidation"], False)
        rendered = json.dumps({"seed": seed_cache, "hit": hit_cache}, sort_keys=True)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)

    def test_openai_cache_replay_canary_decision_includes_request_shape_projection(self):
        rule = cache_module.normalize_cache_pattern_rules([
            {
                "id": "reviewed-openai-cache-replay",
                "enabled": True,
                "policy_source": "managed-recommended",
                "candidate_id": "openai-cache-replay-candidate",
                "conditions": {
                    "pattern_hashes": ["sha256:*"],
                    "source_surface": "openai_responses",
                    "endpoint": "responses",
                    "category": "chat",
                    "has_tools": False,
                    "stream": False,
                },
                "rollout": {
                    "schema": "tokenclaw.pattern_policy_rollout.v1",
                    "recommendation_mode": "canary-only",
                    "canary_enabled": True,
                    "canary_fraction": 1.0,
                    "canary_salt": "openai-cache-replay-test",
                    "canary_unit": "request_fingerprint",
                },
                "action": {
                    "type": "exact_cache_pattern",
                    "allow_tool_calls": False,
                    "safe_invalidation": False,
                    "scope": "session",
                },
                "graduation": {
                "schema": "tokenclaw.openai_cache_replay_shape_activation.v1",
                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                "source_reason": "replay-ready-exact-non-tool-shape",
                "cohort_bucket": "openai_responses/responses/chat/chat/2k_8k_chars/500_2k_tokens",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "chat",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "projected_hits": 55,
                "projected_savings_usd": 0.121981,
                "sample_count": 56,
                "aggregate_only": True,
                "raw_prompt": "raw projection prompt must not leak",
            },
            }
        ])[0]
        rule["rule_id"] = rule["id"]
        rule["canary"] = {
            "schema": "tokenclaw.pattern_canary_decision.v1",
            "enabled": True,
            "selected": True,
            "status": "applied",
            "cohort": "canary_applied",
            "rule_id": rule["id"],
            "candidate_id": "openai-cache-replay-candidate",
            "pattern_hashes": [],
            "raw_pattern_strings_included": False,
        }
        canary_allowed, canary = cache_module.cache_replay_canary_decision(
            cache_meta={"pattern_rule": rule},
            dependency_audit=None,
            session_id="raw-openai-session-must-not-leak",
        )

        self.assertTrue(canary_allowed)
        self.assertEqual(canary["status"], "applied")
        self.assertEqual(canary["policy_source"], "managed-recommended")
        self.assertEqual(canary["cohort_bucket"], "openai_responses/responses/chat/chat/2k_8k_chars/500_2k_tokens")
        self.assertEqual(canary["projected_hits"], 55)
        self.assertEqual(canary["projection"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertFalse(canary["projection"]["raw_request_bodies_included"])
        self.assertFalse(canary["projection"]["cache_keys_included"])
        rendered = json.dumps(canary, sort_keys=True)
        self.assertNotIn("raw projection prompt must not leak", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)

    def test_staged_openai_cache_replay_shape_canary_records_applied_and_holdout_lifecycle(self):
        applied_body = {
            "model": "gpt-5.4-mini",
            "input": " ".join(["Summarize the stable release checklist for this replay canary."] * 90),
        }
        holdout_body = {
            "model": "gpt-5.4-mini",
            "input": " ".join(["Summarize the stable rollout checklist for this replay holdout."] * 90),
        }
        self._enable_staged_openai_cache_replay_shape_rule(canary_fraction=1.0, holdout_fraction=0.0)

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            applied = TestClient(server.app).post(
                "/v1/responses",
                json=applied_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )

        self.assertEqual(applied.status_code, 200)
        [applied_row] = server.store.conn.execute("select cache_hit, cache_json, routing_json from calls").fetchall()
        self.assertEqual(applied_row["cache_hit"], 0)
        applied_cache = json.loads(applied_row["cache_json"])
        applied_routing = json.loads(applied_row["routing_json"])
        self.assertEqual(applied_cache["status"], "miss")
        self.assertEqual(applied_cache["reason"], "exact-pattern-miss")
        self.assertEqual(applied_cache["cache_replay_canary"]["status"], "applied")
        self.assertEqual(applied_cache["cache_replay_canary"]["canary_cohort"], "canary_applied")
        self.assertEqual(applied_cache["cache_replay_canary"]["projection"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(applied_cache["pattern_rule"]["rule_id"], "request-shape-cache-replay-test-policy")
        self.assertEqual(applied_routing["managed_pattern_features"]["token_bucket"], "1k_4k_tokens")
        self.assertNotIn("canary-rule-not-active", json.dumps(applied_cache, sort_keys=True))

        server.store.conn.execute("delete from calls")
        server.store.conn.commit()
        self._enable_staged_openai_cache_replay_shape_rule(canary_fraction=0.0, holdout_fraction=1.0)
        CapturingOpenAIClient.calls = []

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            holdout = TestClient(server.app).post(
                "/v1/responses",
                json=holdout_body,
                headers={"x-session-id": "raw-openai-session-must-not-leak"},
            )

        self.assertEqual(holdout.status_code, 200)
        self.assertGreaterEqual(len(CapturingOpenAIClient.calls), 1)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5.4-mini")
        [holdout_row] = server.store.conn.execute("select cache_hit, cache_json from calls").fetchall()
        self.assertEqual(holdout_row["cache_hit"], 0)
        holdout_cache = json.loads(holdout_row["cache_json"])
        self.assertEqual(holdout_cache["cache_replay_canary"]["status"], "holdout")
        self.assertEqual(holdout_cache["cache_replay_canary"]["canary_cohort"], "canary_holdout")
        self.assertEqual(holdout_cache["cache_replay_canary"]["reason"], "canary_holdout")
        self.assertNotIn("canary-rule-not-active", json.dumps(holdout_cache, sort_keys=True))

    def test_openai_cache_replay_canary_serves_cached_chat_response(self):
        CapturingOpenAIClient.calls = []
        request_body = {
            "model": "gpt-5-codex",
            "messages": [{"role": "user", "content": "Repeatable chat cache replay prompt"}],
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body, path="/v1/chat/completions")
        self._enable_openai_cache_replay_rule(
            pattern_hash,
            source_surface="openai_chat",
            endpoint="chat_completions",
            category="short-completion",
            has_tools=False,
        )
        CapturingOpenAIClient.response_body = {
            "id": "chatcmpl_cached",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "cached chat"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post("/v1/chat/completions", json=request_body, headers={"x-session-id": "chat-session"})
            CapturingOpenAIClient.response_body = {
                "id": "chatcmpl_upstream_should_not_be_used",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "not used"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
            second = TestClient(server.app).post("/v1/chat/completions", json=request_body, headers={"x-session-id": "chat-session"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], "chatcmpl_cached")
        rows = server.store.conn.execute("select cache_hit, stream, cache_json from calls order by rowid").fetchall()
        self.assertEqual(sum(1 for row in rows if row["cache_hit"] == 1), 1)
        self.assertEqual(rows[0]["cache_hit"], 0)
        self.assertEqual(rows[1]["cache_hit"], 1)
        self.assertEqual(rows[0]["stream"], 0)
        self.assertEqual(rows[1]["stream"], 0)
        [hit_row] = [row for row in rows if row["cache_hit"] == 1]
        hit_cache = json.loads(hit_row["cache_json"])
        self.assertEqual(hit_cache["cache_replay_canary"]["reason"], "no-dependency-required")
        self.assertEqual(hit_cache["pattern_rule"]["scope"], "session")

    def test_openai_cache_replay_canary_holdout_forwards_upstream(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "Inspect ./src/holdout.py with the lookup tool.",
            "tools": [{"type": "function", "name": "lookup_file"}],
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body)
        self._enable_openai_cache_replay_rule(pattern_hash, canary_fraction=0.0)

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "holdout-session"})
            second = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "holdout-session"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        rows = server.store.conn.execute("select cache_hit, cache_json from calls order by rowid").fetchall()
        self.assertEqual([row["cache_hit"] for row in rows], [0, 0])
        cache = json.loads(rows[-1]["cache_json"])
        self.assertEqual(cache["status"], "holdout")
        self.assertEqual(cache["reason"], "canary_holdout")
        self.assertEqual(cache["cache_replay_canary"]["status"], "holdout")
        self.assertEqual(cache["cache_replay_canary"]["canary_cohort"], "canary_holdout")

    def test_openai_cache_replay_invalidated_dependency_forwards_upstream(self):
        watched = os.path.join(self.cwd_tmp.name, "src", "changing.py")
        os.makedirs(os.path.dirname(watched), exist_ok=True)
        with open(watched, "w", encoding="utf-8") as f:
            f.write("print('old')\n")
        request_body = {
            "model": "gpt-5-codex",
            "input": "Inspect ./src/changing.py with the lookup tool and summarize it.",
            "tools": [{"type": "function", "name": "lookup_file"}],
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body)
        self._enable_openai_cache_replay_rule(pattern_hash)
        CapturingOpenAIClient.response_body = {
            "id": "resp_old",
            "object": "response",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "old"}]}],
            "usage": {"input_tokens": 25, "output_tokens": 5},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "changing-session"})
            with open(watched, "w", encoding="utf-8") as f:
                f.write("print('new')\n")
            CapturingOpenAIClient.response_body = {
                "id": "resp_new",
                "object": "response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "new"}]}],
                "usage": {"input_tokens": 25, "output_tokens": 5},
            }
            second = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "changing-session"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], "resp_new")
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        rows = server.store.conn.execute("select cache_hit, cache_json from calls order by rowid").fetchall()
        self.assertEqual(rows[-1]["cache_hit"], 0)
        cache = json.loads(rows[-1]["cache_json"])
        self.assertEqual(cache["status"], "invalidated")
        self.assertEqual(cache["reason"], "dependency-changed")
        self.assertEqual(cache["cache_replay_canary"]["status"], "invalidated")
        self.assertFalse(cache["cache_replay_canary"]["dependency_audit"]["paths_included"])

    def test_openai_cache_replay_missing_dependency_evidence_forwards_upstream(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "Inspect ./src/missing.py with the lookup tool and summarize it.",
            "tools": [{"type": "function", "name": "lookup_file"}],
        }
        pattern_hash = self._openai_cache_pattern_hash(request_body)
        self._enable_openai_cache_replay_rule(pattern_hash)

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            first = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "missing-session"})
            second = TestClient(server.app).post("/v1/responses", json=request_body, headers={"x-session-id": "missing-session"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        rows = server.store.conn.execute("select cache_hit, cache_json from calls order by rowid").fetchall()
        self.assertEqual([row["cache_hit"] for row in rows], [0, 0])
        cache = json.loads(rows[-1]["cache_json"])
        self.assertEqual(cache["status"], "bypassed")
        self.assertIn(cache["reason"], {"file-dependency-missing", "dependency-missing"})
        self.assertEqual(cache["cache_replay_store"]["status"], "skipped")
        self.assertFalse(cache["cache_replay_store"]["cache_key_included"])

    def test_openai_route_persists_source_surface_and_sanitized_feature_summaries(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "raw openai prompt",
            "tools": [{"type": "function", "name": "lookup", "description": "raw tool output"}],
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        [row] = server.store.conn.execute(
            """
            select provider, source_surface, endpoint, requested_model_family,
                   routed_model_family, routing_json, request_json, response_json
            from calls
            """
        ).fetchall()
        routing = json.loads(row["routing_json"])
        feature_summary = routing["openai_feature_unit"]
        local_feature_summary = routing["openai_local_feature_unit"]
        outcome_summary = routing["openai_outcome_unit"]

        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["source_surface"], "openai_responses")
        self.assertEqual(row["endpoint"], "responses")
        self.assertEqual(row["requested_model_family"], "gpt-5-codex")
        self.assertEqual(row["routed_model_family"], "gpt-5-codex")
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])
        self.assertEqual(feature_summary["source_surface"], "openai_responses")
        self.assertEqual(feature_summary["endpoint"], "responses")
        self.assertEqual(feature_summary["local_mutation_stage"], "preflight")
        self.assertTrue(feature_summary["has_tools"])
        self.assertFalse(feature_summary["raw_payload_included"])
        self.assertEqual(local_feature_summary["source_surface"], "openai_responses")
        self.assertEqual(local_feature_summary["endpoint"], "responses")
        self.assertEqual(outcome_summary["status_code"], 200)
        self.assertEqual(outcome_summary["actual_input_tokens"], 11)
        self.assertEqual(outcome_summary["actual_output_tokens"], 3)
        self.assertEqual(outcome_summary["source_surface"], "openai_responses")
        self.assertFalse(outcome_summary["raw_payload_included"])
        self.assertNotIn("raw openai prompt", row["routing_json"])
        self.assertNotIn("raw tool output", row["routing_json"])

    def test_openai_tool_light_cache_metadata_captures_dependency_fingerprint_without_raw_paths(self):
        watched = os.path.join(self.cwd_tmp.name, "src", "example.py")
        os.makedirs(os.path.dirname(watched), exist_ok=True)
        with open(watched, "w", encoding="utf-8") as f:
            f.write("print('dependency evidence')\n")
        request_body = {
            "model": "gpt-5-codex",
            "input": "Inspect ./src/example.py and decide whether a lookup tool is needed.",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_file",
                    "description": "raw tool payload must not leak",
                }
            ],
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={"x-session-id": "raw-session-id-must-not-leak"},
            )

        self.assertEqual(response.status_code, 200)
        [row] = server.store.conn.execute("select cache_json, routing_json, request_json from calls").fetchall()
        cache = json.loads(row["cache_json"])
        routing = json.loads(row["routing_json"])
        rendered_cache = json.dumps(cache, sort_keys=True)
        rendered_public = json.dumps({"cache": cache, "routing": routing}, sort_keys=True)

        self.assertEqual(cache["status"], "skipped")
        self.assertEqual(cache["reason"], "tools-disabled")
        self.assertEqual(routing["category"], "tool-light")
        self.assertIn("tool-call-cache-disabled", cache["cache_replay_blocker_reasons"])
        self.assertTrue(cache["file_dependency_evidence_available"])
        self.assertTrue(cache["safe_invalidation_evidence"])
        self.assertEqual(cache["file_dependency_count"], 1)
        self.assertEqual(cache["file_dependency_count_bucket"], "1")
        self.assertTrue(cache["file_dependency_fingerprint_available"])
        self.assertTrue(cache["file_dependency_fingerprint_sha256"].startswith("sha256:"))
        self.assertEqual(cache["file_dependency_fingerprint"]["schema"], "tokenclaw.cache_file_dependency_fingerprint.v1")
        self.assertFalse(cache["file_dependency_fingerprint"]["paths_included"])
        self.assertFalse(cache["file_dependency_fingerprint"]["path_hashes_included"])
        self.assertFalse(cache["file_dependency_audit"]["paths_included"])
        self.assertFalse(cache["file_dependency_audit"]["root_path_included"])
        self.assertIsNone(row["request_json"])
        self.assertNotIn("src/example.py", rendered_cache)
        self.assertNotIn(watched, rendered_cache)
        self.assertNotIn("raw tool payload must not leak", rendered_public)
        self.assertNotIn("raw-session-id-must-not-leak", rendered_public)

    def test_openai_static_cache_metadata_records_missing_dependency_blocker_without_raw_request_body(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "Summarize ./missing-dependency.txt without tools.",
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        [row] = server.store.conn.execute("select cache_json, request_json, response_json from calls").fetchall()
        cache = json.loads(row["cache_json"])
        rendered_cache = json.dumps(cache, sort_keys=True)

        self.assertEqual(cache["status"], "miss")
        self.assertEqual(cache["reason"], "exact-miss")
        self.assertEqual(cache["file_dependency_count"], 1)
        self.assertFalse(cache["file_dependency_evidence_available"])
        self.assertFalse(cache["safe_invalidation_evidence"])
        self.assertFalse(cache["file_dependency_fingerprint"]["paths_included"])
        self.assertIn("dependency-missing", cache["cache_replay_blocker_reasons"])
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])
        self.assertNotIn("missing-dependency.txt", rendered_cache)

    def test_openai_managed_fetch_uses_preflight_unit_before_crunching(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "raw openai prompt",
        }
        events = []

        def fake_crunch(body, *, store_obj=None):
            events.append("crunch")
            crunched = copy.deepcopy(body)
            crunched["input"] = "locally crunched prompt"
            return crunched, {"changed": True, "saved_chars": 7, "status": "applied"}

        async def fake_fetch(unit):
            events.append("fetch")
            self.assertNotIn("crunch", events)
            self.assertEqual(unit["schema"], "tokenclaw.openai_preflight_feature_unit.v1")
            self.assertEqual(unit["input_features"]["local_mutation_stage"], "preflight")
            rendered = json.dumps(unit, sort_keys=True)
            self.assertNotIn("raw openai prompt", rendered)
            self.assertNotIn("locally crunched prompt", rendered)
            return {
                "enabled": True,
                "status": "received",
                "target_model": "gpt-5-mini",
                "confidence": 0.91,
                "policy_id": "openai-mini-candidate",
                "reason": "matched cheap model evidence",
                "optimization_unit_id": 77,
                "policy_source": "managed-recommended",
            }

        with patch.dict(os.environ, {
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "dry-run",
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
        }):
            with patch.object(openai_proxy, "crunch_body", fake_crunch):
                with patch(
                    "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                    side_effect=fake_fetch,
                ):
                    with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                        response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[:2], ["fetch", "crunch"])
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["input"], "locally crunched prompt")
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        self.assertEqual(routing["openai_feature_unit"]["local_mutation_stage"], "preflight")
        self.assertEqual(routing["managed_recommendation"]["mode"], "dry-run")
        self.assertEqual(routing["managed_recommendation"]["status"], "dry-run")

    def test_openai_observe_only_does_not_fetch_or_change_request(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "short prompt",
        }

        with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "1"}, clear=False):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                side_effect=AssertionError("observe-only must not fetch managed recommendations"),
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["mode"], "observe-only")
        self.assertFalse(managed["enabled"])
        self.assertFalse(managed["applied"])

    def test_openai_policy_decision_observe_fetches_predicted_routing_without_changing_request(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "raw openai prompt secret",
        }
        seen = {}

        async def fake_policy_decision(unit, *, request_facts=None):
            seen["unit"] = copy.deepcopy(unit)
            seen["request_facts"] = copy.deepcopy(request_facts)
            rendered = json.dumps(unit, sort_keys=True)
            facts_rendered = json.dumps(request_facts, sort_keys=True)
            self.assertEqual(unit["schema"], "tokenclaw.openai_preflight_feature_unit.v1")
            self.assertEqual(request_facts["schema"], "tokenclaw.request_facts_envelope.v1")
            self.assertNotIn("raw openai prompt secret", rendered)
            self.assertNotIn("raw openai prompt secret", facts_rendered)
            return {
                "enabled": True,
                "status": "received",
                "policy_decision_schema": "tokenclaw.policy_decision.v1",
                "routing_status": "recommended",
                "target_model": "gpt-5-mini",
                "confidence": 0.91,
                "route_down_probability": 0.9,
                "recommended_mode": "shadow",
                "model_artifact_version": "routing-predictor-v1-openai",
                "model_evidence_hash": "sha256:evidence",
                "predictor_rule_id": "routing-evidence:openai:gpt5->mini",
                "routing": {
                    "target_model": "gpt-5-mini",
                    "route_proposal": {
                        "traffic_treatment": "shadow",
                        "route_selected": False,
                    },
                },
                "policy_id": "openai-predictor-policy",
                "reason": "active predictor",
                "policy_source": "managed-recommended",
            }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_POLICY_DECISION_ENABLED": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_policy_decision",
                side_effect=fake_policy_decision,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["status"], "held")
        self.assertEqual(managed["apply_reason"], "observe-only-local-gate")
        self.assertEqual(managed["would_route_model"], "gpt-5-mini")
        self.assertFalse(managed["applied"])
        self.assertIn("unit", seen)
        self.assertNotIn("raw openai prompt secret", row["routing_json"])

    def test_openai_policy_decision_server_shadow_runs_without_local_sampling(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "raw canary prompt secret",
        }
        seen = {}

        async def fake_policy_decision(unit, *, request_facts=None):
            seen["unit"] = copy.deepcopy(unit)
            seen["request_facts"] = copy.deepcopy(request_facts)
            rendered = json.dumps(unit, sort_keys=True)
            facts_rendered = json.dumps(request_facts, sort_keys=True)
            self.assertEqual(unit["schema"], "tokenclaw.openai_preflight_feature_unit.v1")
            self.assertEqual(request_facts["schema"], "tokenclaw.request_facts_envelope.v1")
            self.assertNotIn("raw canary prompt secret", rendered)
            self.assertNotIn("raw canary prompt secret", facts_rendered)
            return {
                "enabled": True,
                "status": "received",
                "policy_decision_schema": "tokenclaw.policy_decision.v1",
                "routing_status": "recommended",
                "target_model": "gpt-5-mini",
                "confidence": 0.91,
                "route_down_probability": 0.9,
                "recommended_mode": "shadow",
                "model_artifact_version": "routing-predictor-v1-openai",
                "model_evidence_hash": "sha256:evidence",
                "predictor_rule_id": "routing-evidence:openai:gpt5->mini",
                "routing": {
                    "target_model": "gpt-5-mini",
                    "route_proposal": {
                        "traffic_treatment": "shadow",
                        "route_selected": False,
                    },
                },
                "policy_id": "openai-predictor-policy",
                "reason": "active predictor",
                "policy_source": "managed-recommended",
            }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_POLICY_DECISION_ENABLED": "1",
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_policy_decision",
                side_effect=fake_policy_decision,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        self.assertEqual(CapturingOpenAIClient.calls[1]["json"]["model"], "gpt-5-mini")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["mode"], "policy-decision-shadow")
        self.assertEqual(managed["status"], "held")
        self.assertFalse(managed["applied"])
        self.assertFalse(managed["changed_model"])
        self.assertTrue(managed["shadow_only"])
        self.assertEqual(managed["shadow_model"], "gpt-5-mini")
        self.assertEqual(managed["would_route_model"], "gpt-5-mini")
        self.assertEqual(managed["server_traffic_treatment"], "shadow")
        self.assertEqual(routing["managed_route_candidate_model"], "gpt-5-mini")
        self.assertTrue(routing["managed_route_shadow_only"])
        self.assertEqual(routing["routing_experiment"]["trigger"], "managed-policy-routing-canary")
        self.assertTrue(routing["routing_experiment"]["shadow_only"])
        self.assertIn("unit", seen)
        self.assertNotIn("raw canary prompt secret", row["routing_json"])

    def test_openai_policy_decision_canary_holdout_preserves_local_model(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "holdout prompt",
        }

        async def fake_policy_decision(_unit, *, request_facts=None):
            self.assertEqual(request_facts["schema"], "tokenclaw.request_facts_envelope.v1")
            return {
                "enabled": True,
                "status": "received",
                "policy_decision_schema": "tokenclaw.policy_decision.v1",
                "routing_status": "recommended",
                "target_model": "gpt-5-mini",
                "confidence": 0.91,
                "route_down_probability": 0.9,
                "recommended_mode": "canary",
                "model_artifact_version": "routing-predictor-v1-openai",
                "model_evidence_hash": "sha256:evidence",
                "predictor_rule_id": "routing-evidence:openai:gpt5->mini",
                "routing": {
                    "target_model": "gpt-5-mini",
                    "route_proposal": {
                        "traffic_treatment": "holdout",
                        "route_selected": False,
                    },
                },
                "policy_id": "openai-predictor-policy",
                "reason": "active predictor",
                "policy_source": "managed-recommended",
            }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_POLICY_DECISION_ENABLED": "1",
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_policy_decision",
                side_effect=fake_policy_decision,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["status"], "held")
        self.assertEqual(managed["apply_reason"], "server-canary-holdout")
        self.assertEqual(managed["server_traffic_treatment"], "holdout")
        self.assertFalse(managed["applied"])

    def test_openai_dry_run_recommendation_records_projection_without_changing_request(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "short prompt",
        }
        recommendation = {
            "enabled": True,
            "status": "received",
            "target_model": "gpt-5-mini",
            "confidence": 0.91,
            "policy_id": "openai-mini-candidate",
            "reason": "matched cheap model evidence",
            "optimization_unit_id": 77,
            "matched_sample_count": 42,
            "error_rate": 0.01,
            "retry_rate": 0.02,
            "fallback_rate": 0.03,
            "latency_regression_ratio": 0.8,
            "policy_source": "managed-recommended",
        }

        with patch.dict(os.environ, {"TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "dry-run"}):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["mode"], "dry-run")
        self.assertEqual(managed["status"], "dry-run")
        self.assertFalse(managed["applied"])
        self.assertTrue(managed["would_change_model"])
        self.assertEqual(managed["would_route_model"], "gpt-5-mini")
        self.assertEqual(managed["projection"]["matched_sample_count"], 42)
        self.assertEqual(managed["projection"]["risk"]["error_rate"], 0.01)
        self.assertIsNotNone(managed["projection"]["current_input_cost_est_usd"])
        self.assertIsNotNone(managed["projection"]["target_input_cost_est_usd"])

    def test_openai_canary_shadows_only_selected_safe_openai_model(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "short prompt",
        }
        recommendation = {
            "enabled": True,
            "status": "received",
            "target_model": "gpt-5-mini",
            "confidence": 0.91,
            "policy_id": "openai-mini-candidate",
            "reason": "matched cheap model evidence",
            "optimization_unit_id": 77,
            "policy_source": "managed-recommended",
        }

        with patch.dict(os.environ, {
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 1)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["mode"], "canary")
        self.assertEqual(managed["status"], "skipped")
        self.assertEqual(managed["apply_reason"], "server-action-state-missing")
        self.assertFalse(managed["applied"])
        self.assertFalse(managed["changed_model"])
        self.assertNotIn("managed_route_candidate_model", routing)

    def test_openai_bad_recommendation_falls_back_to_local_request(self):
        request_body = {
            "model": "gpt-5-codex",
            "input": "short prompt",
        }
        recommendation = {
            "enabled": True,
            "status": "received",
            "target_model": "claude-haiku-4-5-20251001",
            "confidence": 0.91,
            "policy_id": "wrong-provider-candidate",
            "reason": "bad provider",
            "optimization_unit_id": 77,
            "policy_source": "managed-recommended",
        }

        with patch.dict(os.environ, {
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(managed["status"], "skipped")
        self.assertEqual(managed["apply_reason"], "provider-mismatch")
        self.assertFalse(managed["applied"])

    def test_openai_policy_decision_applies_routing_crunch_and_cache_profiles_locally(self):
        long_text = "alpha " * 2200
        request_body = {
            "model": "gpt-5-codex",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "finish"},
            ],
        }
        recommendation = {
            "schema": "tokenclaw.policy_decision.v1",
            "enabled": True,
            "status": "received",
            "provider": "openai",
            "source_surface": "openai_chat",
            "confidence": 0.94,
            "policy_id": "managed-local-actions",
            "reason": "feature-only policy decision",
            "policy_decision_schema": "tokenclaw.policy_decision.v1",
            "routing_status": "recommended",
            "recommended_mode": "live",
            "route_down_probability": 0.9,
            "model_artifact_version": "routing-predictor-v1-openai",
            "model_evidence_hash": "sha256:evidence",
            "predictor_rule_id": "routing-evidence:openai:gpt5->mini",
            "routing": {"target_model": "gpt-5-mini"},
            "crunch": {"profile": "aggressive", "threshold_chars": 1000},
            "cache": {"profile": "semantic", "semantic_threshold": 0.82},
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/chat/completions", json=request_body)

        self.assertEqual(response.status_code, 200)
        forwarded = CapturingOpenAIClient.calls[0]["json"]
        self.assertEqual(forwarded["model"], "gpt-5-mini")
        self.assertIn("middle of long older text block omitted", forwarded["messages"][1]["content"])
        [row] = server.store.conn.execute("select routed_model, routing_json, crunch_json, cache_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        crunch = json.loads(row["crunch_json"])
        cache = json.loads(row["cache_json"])

        self.assertEqual(row["routed_model"], "gpt-5-mini")
        self.assertEqual(routing["managed_recommendation"]["status"], "applied")
        self.assertTrue(routing["managed_recommendation"]["changed_model"])
        self.assertEqual(routing["managed_local_actions"]["crunch"]["status"], "applied")
        self.assertEqual(routing["managed_local_actions"]["cache"]["status"], "applied")
        self.assertEqual(crunch["policy_source"], "managed-recommended")
        self.assertEqual(crunch["threshold_chars"], 1000)
        self.assertEqual(cache["policy_source"], "managed-recommended")
        self.assertTrue(cache["semantic_enabled"])
        self.assertEqual(cache["semantic_threshold"], 0.82)
        self.assertNotIn(long_text, row["routing_json"])

    def test_openai_old_context_summary_canary_applies_before_forwarding(self):
        old_text = "old openai secret context should not be logged or forwarded. " * 120
        request_body = {
            "model": "gpt-5.4",
            "instructions": "developer instruction remains top-level",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": old_text + "one"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": old_text + "two"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "recent user stays"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "recent assistant stays"}]},
            ],
        }
        policy_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        policy_file.write("\n".join([
            "openai_old_context_summarization:",
            "  enabled: true",
            "  rule_id: test-openai-summary-canary",
            "  summary_model: gpt-5-mini",
            "  min_request_chars: 0",
            "  min_source_chars: 1",
            "  keep_recent_items: 2",
            "  max_summary_chars: 200",
            "  max_summary_cost_usd: 1",
            "  blocked_categories: []",
            "  canary:",
            "    enabled: true",
            "    canary_fraction: 1",
            "    holdout_fraction: 0",
            "    salt: route-test",
            "rules: []",
            "",
        ]))
        policy_file.close()
        self.addCleanup(lambda: os.path.exists(policy_file.name) and os.unlink(policy_file.name))
        CapturingOpenAIClient.calls = []
        CapturingOpenAIClient.response_body = {
            "id": "resp_summary_or_final",
            "object": "response",
            "model": "gpt-5-mini",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "safe compact continuity summary"}],
                }
            ],
            "usage": {"input_tokens": 40, "output_tokens": 8},
        }

        with patch.dict(os.environ, {"TOKENCLAW_CRUNCH_RULES": policy_file.name}, clear=False):
            with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        summary_request = CapturingOpenAIClient.calls[0]["json"]
        forwarded = CapturingOpenAIClient.calls[1]["json"]
        self.assertEqual(summary_request["model"], "gpt-5-mini")
        self.assertIn("old openai secret context", summary_request["input"])
        forwarded_json = json.dumps(forwarded, sort_keys=True)
        self.assertIn("safe compact continuity summary", forwarded_json)
        self.assertIn("recent user stays", forwarded_json)
        self.assertIn("recent assistant stays", forwarded_json)
        self.assertNotIn("old openai secret context", forwarded_json)

        [row] = server.store.conn.execute("select crunch_json, cost_est_usd from calls").fetchall()
        crunch = json.loads(row["crunch_json"])
        summary = crunch["old_context_summarization"]
        self.assertEqual(summary["status"], "applied")
        self.assertTrue(summary["applied"])
        self.assertEqual(summary["canary"]["cohort"], "canary_applied")
        self.assertFalse(summary["summary_cache_hit"])
        self.assertGreater(summary["summary_cost_est_usd"], 0)
        self.assertGreater(summary["estimated_tokens_saved"], 0)
        self.assertGreater(row["cost_est_usd"], summary["summary_cost_est_usd"])
        rendered_meta = json.dumps(summary, sort_keys=True)
        self.assertNotIn("safe compact continuity summary", rendered_meta)
        self.assertNotIn("old openai secret context", rendered_meta)

    def test_openai_policy_decision_rejects_raw_like_unsafe_actions(self):
        request_body = {"model": "gpt-5-codex", "input": "short prompt"}
        recommendation = {
            "schema": "tokenclaw.policy_decision.v1",
            "enabled": True,
            "status": "received",
            "provider": "openai",
            "source_surface": "openai_responses",
            "confidence": 0.9,
            "policy_id": "unsafe-policy",
            "reason": "unsafe",
            "routing": {"target_model": "gpt-5-mini"},
            "crunch": {"profile": "aggressive", "threshold_chars": 1},
            "privacy_summary": {"metadata_only": False, "raw_payload_included": True},
        }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routing_json, crunch_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        crunch = json.loads(row["crunch_json"])
        self.assertEqual(routing["managed_recommendation"]["status"], "vetoed")
        self.assertEqual(routing["managed_recommendation"]["apply_reason"], "privacy-not-metadata-only")
        self.assertEqual(crunch["policy_source"], "local-default")

    def test_openai_enhanced_crunch_hint_uses_configured_local_provider(self):
        request_body = {
            "model": "gpt-5-codex",
            "messages": [
                {"role": "user", "content": "raw openai prompt SECRET_ENHANCED_CRUNCH"},
            ],
        }
        recommendation = {
            "schema": "tokenclaw.policy_decision.v1",
            "enabled": True,
            "status": "received",
            "provider": "openai",
            "source_surface": "openai_chat",
            "confidence": 0.9,
            "policy_id": "enhanced-crunch-policy",
            "reason": "feature-only enhanced crunch hint",
            "crunch": {
                "profile": "old_context_summarization",
                "old_context_summarization": {
                    "enabled": True,
                    "model_family": "haiku",
                    "thresholds": {"min_request_chars": 10, "min_summarized_chars": 10},
                    "max_summary_cost_usd": 0.002,
                    "canary": {"enabled": True, "fraction": 0.25},
                    "safety_stop": {"max_error_rate": 0.05},
                },
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }

        with patch.dict(os.environ, {
            "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/chat/completions", json=request_body)

        self.assertEqual(response.status_code, 200)
        [row] = server.store.conn.execute("select routing_json, crunch_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        crunch = json.loads(row["crunch_json"])
        action = routing["managed_local_actions"]["crunch"]

        self.assertEqual(action["status"], "configured")
        self.assertEqual(action["old_context_summarization"]["state"], "configured")
        self.assertEqual(action["enhanced_crunch"]["state"], "configured")
        self.assertEqual(crunch["policy_source"], "managed-recommended")
        self.assertNotIn("SECRET_ENHANCED_CRUNCH", row["routing_json"])

    def test_openai_policy_decision_rejects_expired_provider_mismatch_and_unknown_actions(self):
        cases = [
            (
                {
                    "schema": "tokenclaw.policy_decision.v1",
                    "enabled": True,
                    "status": "received",
                    "provider": "openai",
                    "source_surface": "openai_responses",
                    "expires_at": "2000-01-01T00:00:00Z",
                    "confidence": 0.9,
                    "policy_id": "expired-policy",
                    "reason": "expired",
                    "routing": {"target_model": "gpt-5-mini"},
                },
                "expired-policy",
            ),
            (
                {
                    "schema": "tokenclaw.policy_decision.v1",
                    "enabled": True,
                    "status": "received",
                    "provider": "anthropic",
                    "source_surface": "openai_responses",
                    "confidence": 0.9,
                    "policy_id": "provider-mismatch-policy",
                    "reason": "wrong provider",
                    "routing": {"target_model": "gpt-5-mini"},
                },
                "provider-mismatch",
            ),
            (
                {
                    "schema": "tokenclaw.policy_decision.v1",
                    "enabled": True,
                    "status": "received",
                    "provider": "openai",
                    "source_surface": "openai_responses",
                    "confidence": 0.9,
                    "policy_id": "unknown-action-policy",
                    "reason": "unknown action",
                    "actions": [{"type": "replacement_prompt"}],
                    "routing": {"target_model": "gpt-5-mini"},
                },
                "unsupported-action-type",
            ),
        ]
        for index, (recommendation, expected) in enumerate(cases):
            CapturingOpenAIClient.calls = []
            with self.subTest(expected=expected):
                with patch.dict(os.environ, {
                    "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
                    "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
                    "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
                }):
                    with patch(
                        "tokenclaw.optimization.openai_recommendations.fetch_recommendation",
                        return_value=recommendation,
                    ):
                        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                            response = TestClient(server.app).post(
                                "/v1/responses",
                                json={"model": "gpt-5-codex", "input": f"short prompt {index}"},
                            )
                self.assertEqual(response.status_code, 200)
                routing = json.loads(
                    server.store.conn.execute(
                        "select routing_json from calls order by created_at desc limit 1"
                    ).fetchone()["routing_json"]
                )
                managed = routing["managed_recommendation"]
                if expected == "unsupported-action-type":
                    self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
                    self.assertEqual(
                        managed["local_actions"]["unsupported_actions"][0]["reason"],
                        "unsupported-action-type",
                    )
                    self.assertEqual(managed["status"], "vetoed")
                    self.assertEqual(managed["apply_reason"], "unsupported-action-type")
                    self.assertFalse(managed["changed_model"])
                else:
                    self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
                    self.assertEqual(managed["status"], "vetoed")
                    self.assertEqual(managed["apply_reason"], expected)

    def test_openai_policy_decision_falls_back_when_managed_disabled(self):
        with patch.dict(os.environ, {
            "TOKENCLAW_OPENAI_RECOMMENDATION_MODE": "canary",
            "TOKENCLAW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }, clear=False):
            os.environ.pop("TOKENCLAW_RECOMMENDATION_ENABLED", None)
            with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                response = TestClient(server.app).post(
                    "/v1/responses",
                    json={"model": "gpt-5-codex", "input": "short prompt"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        managed = json.loads(row["routing_json"])["managed_recommendation"]
        self.assertEqual(managed["status"], "skipped")
        self.assertEqual(managed["apply_reason"], "disabled")
        self.assertFalse(managed["applied"])

    @unittest.skip("local OpenAI canary routing retired; managed policy decisions own canary/shadow treatment")
    def test_openai_local_canary_shadows_in_responses_optimized_path(self):
        self._enable_openai_canary(canary_fraction=1.0)
        request_body = {"model": "gpt-5-codex", "input": "short prompt SECRET_OPENAI_CANARY"}

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        self.assertEqual(CapturingOpenAIClient.calls[1]["json"]["model"], "gpt-5-mini")
        [row] = server.store.conn.execute(
            "select requested_model, routed_model, routing_json, request_json from calls"
        ).fetchall()
        routing = json.loads(row["routing_json"])
        canary = routing["openai_canary"]

        self.assertEqual(row["requested_model"], "gpt-5-codex")
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertIsNone(row["request_json"])
        self.assertEqual(canary["status"], "applied")
        self.assertEqual(canary["cohort"], "canary_applied")
        self.assertEqual(canary["rule_id"], "test-openai-local-canary")
        self.assertEqual(canary["candidate_id"], "test-openai-candidate")
        self.assertEqual(canary["original_model"], "gpt-5-codex")
        self.assertEqual(canary["target_model"], "gpt-5-mini")
        self.assertEqual(canary["actual_forwarded_model"], "gpt-5-codex")
        self.assertEqual(canary["shadow_model"], "gpt-5-mini")
        self.assertTrue(canary["shadow_only"])
        self.assertTrue(canary["requires_shadow_comparison"])
        self.assertTrue(canary["live_promotion_required"])
        self.assertEqual(routing["routing_experiment"]["trigger"], "openai-local-routing-canary")
        self.assertEqual(routing["routing_experiment"]["primary_model"], "gpt-5-codex")
        self.assertEqual(routing["routing_experiment"]["shadow_model"], "gpt-5-mini")
        self.assertTrue(routing["routing_experiment"]["shadow_only"])
        self.assertTrue(canary["cohort_key_hash"].startswith("sha256:"))
        self.assertIsNotNone(canary["projected_input_savings_usd"])
        self.assertNotIn("SECRET_OPENAI_CANARY", row["routing_json"])

    @unittest.skip("local OpenAI canary routing retired; managed policy decisions own canary/holdout treatment")
    def test_openai_local_canary_holdout_keeps_chat_requested_model(self):
        self._enable_openai_canary(canary_fraction=0.0, holdout_fraction=1.0)
        request_body = {
            "model": "gpt-5-codex",
            "messages": [{"role": "user", "content": "short chat prompt"}],
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post("/v1/chat/completions", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        canary = json.loads(row["routing_json"])["openai_canary"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(canary["status"], "holdout")
        self.assertEqual(canary["cohort"], "canary_holdout")
        self.assertEqual(canary["actual_forwarded_model"], "gpt-5-codex")

    @unittest.skip("local OpenAI canary routing retired; managed server owns cohort coverage")
    def test_openai_gpt54_canary_sequence_restores_applied_and_holdout_report_coverage(self):
        os.environ["TOKENCLAW_DB"] = self.tmp.name
        self._enable_openai_canary(
            canary_fraction=0.15,
            holdout_fraction=0.10,
            target_model="gpt-5.4-mini",
            model_pattern="gpt-5.4",
        )
        CapturingOpenAIClient.calls = []

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            for idx in range(22):
                response = TestClient(server.app).post(
                    "/v1/responses",
                    json={"model": "gpt-5.4", "input": f"short prompt SECRET_OPENAI_CANARY_{idx}"},
                )
                self.assertEqual(response.status_code, 200)

        rows = server.store.conn.execute("select routed_model, routing_json from calls order by created_at").fetchall()
        canaries = [json.loads(row["routing_json"])["openai_canary"] for row in rows]
        applied = [canary for canary in canaries if canary["cohort"] == "canary_applied"]
        holdout = [canary for canary in canaries if canary["cohort"] == "canary_holdout"]

        self.assertGreater(len(applied), 0)
        self.assertGreater(len(holdout), 0)
        self.assertTrue(all(canary["requested_model"] == "gpt-5.4" for canary in canaries))
        self.assertTrue(all(canary["target_model"] == "gpt-5.4-mini" for canary in canaries))
        self.assertTrue(all(canary["cohort_unit"] == "request_features_sequence" for canary in canaries))
        self.assertTrue(all("cohort_sequence_index" in canary for canary in canaries))
        self.assertTrue(all(canary["cohort_features"]["requested_model"] == "gpt-5.4" for canary in canaries))
        self.assertTrue(all(row["routed_model"] == "gpt-5.4" for row in rows))
        self.assertTrue(any(canary.get("shadow_model") == "gpt-5.4-mini" for canary in applied))

        report = build_openai_routing_report(server.store, limit=50)
        candidate = next(row for row in report["candidates"] if row["requested_model"] == "gpt-5.4")
        lifecycle = candidate["openai_canary_lifecycle_evidence"]

        self.assertGreaterEqual(lifecycle["cohort_counts"]["canary_applied"], 3)
        self.assertGreaterEqual(lifecycle["cohort_counts"]["canary_holdout"], 2)
        self.assertEqual(lifecycle["blocker_codes"], [])
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)
        self.assertGreater(report["summary"]["openai_canary_applied_count"], 0)
        self.assertGreater(report["summary"]["openai_canary_holdout_count"], 0)
        self.assertGreater(report["summary"]["estimated_savings_per_1000_calls_usd"], 0)

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("SECRET_OPENAI_CANARY", rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(lifecycle["privacy"]["request_ids_included"])
        self.assertFalse(lifecycle["privacy"]["session_ids_included"])
        self.assertFalse(lifecycle["privacy"]["cache_keys_included"])

    @unittest.skip("local OpenAI canary routing retired; managed server owns target feasibility")
    def test_openai_local_canary_records_incompatible_target_noop(self):
        self._enable_openai_canary(canary_fraction=1.0, target_model="claude-haiku-4-5-20251001")

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json={"model": "gpt-5-codex", "input": "short prompt"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        canary = json.loads(row["routing_json"])["openai_canary"]

        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(canary["status"], "ineligible")
        self.assertEqual(canary["reason"], "provider-mismatch")

    @unittest.skip("local OpenAI canary routing retired; managed policy decisions own stream treatment")
    def test_openai_local_canary_records_streaming_metadata_when_allowed(self):
        self._enable_openai_canary(canary_fraction=1.0, allow_stream=True)
        CapturingStreamingOpenAIClient.stream_calls = []

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingStreamingOpenAIClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json={"model": "gpt-5-codex", "input": "short prompt", "stream": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingStreamingOpenAIClient.stream_calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select stream, routed_model, routing_json from calls").fetchall()
        canary = json.loads(row["routing_json"])["openai_canary"]

        self.assertEqual(row["stream"], 1)
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertEqual(canary["status"], "applied")
        self.assertEqual(canary["actual_forwarded_model"], "gpt-5-codex")
        self.assertEqual(canary["shadow_model"], "gpt-5-mini")
        self.assertTrue(canary["stream"])
        self.assertFalse(canary["requires_shadow_comparison"])

    @unittest.skip("local OpenAI canary routing retired; managed policy decisions own fallback treatment")
    def test_openai_local_canary_rate_limit_fallback_records_requested_model(self):
        self._enable_openai_canary(canary_fraction=1.0)
        RateLimitThenSuccessOpenAIClient.calls = []
        RateLimitThenSuccessOpenAIClient.status_code = 200
        RateLimitThenSuccessOpenAIClient.response_body = CapturingOpenAIClient.response_body

        async def no_sleep(delay):
            return None

        with patch.object(openai_proxy.asyncio, "sleep", new=no_sleep):
            with patch.object(openai_proxy.httpx, "AsyncClient", RateLimitThenSuccessOpenAIClient):
                response = TestClient(server.app).post(
                    "/v1/responses",
                    json={"model": "gpt-5-codex", "input": "short prompt"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RateLimitThenSuccessOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        self.assertEqual(RateLimitThenSuccessOpenAIClient.calls[1]["json"]["model"], "gpt-5-codex")
        self.assertEqual(RateLimitThenSuccessOpenAIClient.calls[2]["json"]["model"], "gpt-5-mini")
        [row] = server.store.conn.execute("select retry_count, routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        canary = routing["openai_canary"]

        self.assertEqual(row["retry_count"], 1)
        self.assertEqual(row["routed_model"], "gpt-5-codex")
        self.assertNotIn("fallback_reason", routing)
        self.assertEqual(canary["status"], "applied")
        self.assertEqual(canary["actual_forwarded_model"], "gpt-5-codex")

    @unittest.skip("local OpenAI canary-triggered shadow experiments retired")
    def test_openai_routing_experiment_records_primary_shadow_metadata_only(self):
        self._enable_openai_canary(canary_fraction=1.0)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as policy_file:
            policy_file.write("\n".join([
                "profile_id: first-safe-openai-codex-ab-v1",
                "mode: applied_routed_down",
                "enabled: true",
                "sample_rate: 1.0",
                "daily_budget_usd: 0.05",
                "providers:",
                "  - openai",
                "source_surfaces:",
                "  - openai_responses",
                "model_pairs:",
                "  - requested_model: gpt-5-codex",
                "    routed_model: gpt-5-mini",
                "categories:",
                "  - short-completion",
                "  - chat",
                "max_text_chars: 8000",
                "store_response_bodies: false",
                "",
            ]))
            policy_path = policy_file.name
        self.addCleanup(lambda: os.path.exists(policy_path) and os.unlink(policy_path))
        os.environ["TOKENCLAW_ROUTING_EXPERIMENTS"] = policy_path
        importlib.reload(routing_experiments_module)
        CapturingOpenAIClient.calls = []
        CapturingOpenAIClient.status_code = 200
        CapturingOpenAIClient.response_body = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5-mini",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "experiment ok"}],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json={"model": "gpt-5-codex", "input": "short prompt SECRET_OPENAI_AB"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        self.assertEqual(CapturingOpenAIClient.calls[1]["json"]["model"], "gpt-5-mini")

        [experiment_row] = server.store.conn.execute(
            """
            select provider, source_surface, requested_model, routed_model,
                   primary_model, shadow_model, primary_status_code, shadow_status_code,
                   output_similarity, primary_response_json, shadow_response_json, experiment_json
            from routing_experiments
            """
        ).fetchall()
        experiment = json.loads(experiment_row["experiment_json"])
        [call_row] = server.store.conn.execute(
            "select routing_json, request_json, response_json from calls"
        ).fetchall()
        routing = json.loads(call_row["routing_json"])
        queue_row = server.store.conn.execute(
            "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue "
            "where source_surface = 'routing_experiment_outcome'"
        ).fetchone()

        self.assertEqual(experiment_row["provider"], "openai")
        self.assertEqual(experiment_row["source_surface"], "openai_responses")
        self.assertEqual(experiment_row["requested_model"], "gpt-5-codex")
        self.assertEqual(experiment_row["routed_model"], "gpt-5-mini")
        self.assertEqual(experiment_row["primary_model"], "gpt-5-codex")
        self.assertEqual(experiment_row["shadow_model"], "gpt-5-mini")
        self.assertEqual(experiment_row["primary_status_code"], 200)
        self.assertEqual(experiment_row["shadow_status_code"], 200)
        self.assertEqual(experiment_row["output_similarity"], 1.0)
        self.assertIsNone(experiment_row["primary_response_json"])
        self.assertIsNone(experiment_row["shadow_response_json"])
        self.assertEqual(experiment["status"], "compared")
        self.assertEqual(experiment["managed_feedback"]["status"], "queued")
        self.assertEqual(routing["routing_experiment"]["status"], "compared")
        self.assertEqual(routing["routing_experiment"]["profile_id"], "first-safe-openai-codex-ab-v1")
        self.assertIsNone(call_row["request_json"])
        self.assertIsNone(call_row["response_json"])
        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        self.assertEqual(queue_row["endpoint"], "/v1/policy-events")
        self.assertEqual(queue_row["status"], "queued")
        self.assertNotIn("SECRET_OPENAI_AB", queue_row["payload_json"])
        self.assertNotIn("SECRET_OPENAI_AB", call_row["routing_json"])

    def test_openai_shadow_candidate_pass_through_keeps_primary_requested_model(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as policy_file:
            policy_file.write("\n".join([
                "profile_id: first-safe-openai-codex-ab-v1",
                "mode: shadow_candidate_pass_through",
                "enabled: true",
                "sample_rate: 1.0",
                "daily_budget_usd: 10.0",
                "providers:",
                "  - openai",
                "source_surfaces:",
                "  - openai_responses",
                "model_pairs:",
                "  - requested_model: gpt-5.4",
                "    routed_model: gpt-5.4-mini",
                "categories:",
                "  - short-completion",
                "  - chat",
                "max_text_chars: 8000",
                "store_response_bodies: false",
                "",
            ]))
            policy_path = policy_file.name
        self.addCleanup(lambda: os.path.exists(policy_path) and os.unlink(policy_path))
        os.environ["TOKENCLAW_ROUTING_EXPERIMENTS"] = policy_path
        importlib.reload(routing_experiments_module)
        CapturingOpenAIClient.calls = []
        CapturingOpenAIClient.status_code = 200
        CapturingOpenAIClient.response_body = {
            "id": "resp_primary",
            "object": "response",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "pass through ok"}],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

        with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json={"model": "gpt-5.4", "input": "short prompt SECRET_SHADOW_AB"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "gpt-5.4")
        self.assertEqual(len(CapturingOpenAIClient.calls), 2)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5.4")
        self.assertEqual(CapturingOpenAIClient.calls[1]["json"]["model"], "gpt-5.4-mini")

        [experiment_row] = server.store.conn.execute(
            """
            select provider, source_surface, requested_model, routed_model,
                   primary_model, shadow_model, primary_status_code, shadow_status_code,
                   primary_response_json, shadow_response_json, experiment_json
            from routing_experiments
            """
        ).fetchall()
        experiment = json.loads(experiment_row["experiment_json"])
        [call_row] = server.store.conn.execute(
            "select requested_model, routed_model, routing_json, request_json, response_json from calls"
        ).fetchall()
        routing = json.loads(call_row["routing_json"])
        queue_row = server.store.conn.execute(
            "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue "
            "where source_surface = 'routing_experiment_outcome'"
        ).fetchone()
        payload = json.loads(queue_row["payload_json"])

        self.assertEqual(call_row["requested_model"], "gpt-5.4")
        self.assertEqual(call_row["routed_model"], "gpt-5.4")
        self.assertEqual(experiment_row["requested_model"], "gpt-5.4")
        self.assertEqual(experiment_row["routed_model"], "gpt-5.4-mini")
        self.assertEqual(experiment_row["primary_model"], "gpt-5.4")
        self.assertEqual(experiment_row["shadow_model"], "gpt-5.4-mini")
        self.assertEqual(experiment_row["primary_status_code"], 200)
        self.assertEqual(experiment_row["shadow_status_code"], 200)
        self.assertIsNone(experiment_row["primary_response_json"])
        self.assertIsNone(experiment_row["shadow_response_json"])
        self.assertEqual(experiment["mode"], "shadow_candidate_pass_through")
        self.assertTrue(experiment["counterfactual"])
        self.assertTrue(experiment["shadow_only"])
        self.assertEqual(experiment["reason"], "sampled-shadow-candidate-pass-through")
        self.assertEqual(routing["routing_experiment"]["primary_model"], "gpt-5.4")
        self.assertEqual(routing["routing_experiment"]["shadow_model"], "gpt-5.4-mini")
        self.assertEqual(payload["candidate"]["mode"], "shadow_candidate_pass_through")
        self.assertTrue(payload["candidate"]["counterfactual"])
        self.assertTrue(payload["candidate"]["shadow_only"])
        self.assertNotIn("SECRET_SHADOW_AB", queue_row["payload_json"])
        self.assertNotIn("SECRET_SHADOW_AB", call_row["routing_json"])
        self.assertIsNone(call_row["request_json"])
        self.assertIsNone(call_row["response_json"])


if __name__ == "__main__":
    unittest.main()
