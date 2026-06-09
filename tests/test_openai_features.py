import copy
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy.optimization import openai_features
from agentflow_proxy.store import stable_json


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import openai_proxy, server
    from agentflow_proxy.store import Store


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
        self._assert_no_raw_values(unit)
        self._assert_no_raw_values(summary)

    def test_preflight_unit_is_feature_only_before_local_mutation(self):
        unit = openai_features.build_openai_preflight_feature_unit(
            body={
                "model": "gpt-5-codex",
                "input": [
                    {"role": "system", "content": "raw openai prompt"},
                    {"type": "message", "content": "raw chat message /home/lutz/project/app.py"},
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

        self.assertEqual(unit["schema"], "agentflow.openai_preflight_feature_unit.v1")
        self.assertEqual(unit["source_surface"], "openai_responses")
        self.assertEqual(unit["candidate_target_model"], None)
        self.assertEqual(unit["grouping_identifiers"], {})
        self.assertEqual(unit["input_features"]["local_mutation_stage"], "preflight")
        self.assertEqual(unit["input_features"]["path_class"], "responses")
        self.assertEqual(unit["input_features"]["old_context"]["shape"], "responses_input_items")
        self.assertTrue(unit["input_features"]["cache_eligibility"]["streaming_bypass_hint"])
        self.assertFalse(unit["input_features"]["cache_eligibility"]["raw_cache_key_included"])
        self.assertEqual(summary["local_mutation_stage"], "preflight")
        self.assertTrue(summary["has_tools"])
        for forbidden_key in ("messages", "input", "cache_key", "request_id", "session_id"):
            self.assertNotIn(f'"{forbidden_key}"', rendered)
        self._assert_no_raw_values(unit)
        self.assertNotIn("/home/lutz/project/app.py", rendered)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class OpenAIFeatureRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.old_log_bodies = server.LOG_BODIES
        self.saved_recommendation_enabled = os.environ.get("AGENTFLOW_RECOMMENDATION_ENABLED")
        os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
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
            os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
        else:
            os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = self.saved_recommendation_enabled
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
            self.assertEqual(unit["schema"], "agentflow.openai_preflight_feature_unit.v1")
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
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "dry-run",
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
        }):
            with patch.object(openai_proxy, "crunch_body", fake_crunch):
                with patch(
                    "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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

        with patch.dict(os.environ, {"AGENTFLOW_RECOMMENDATION_ENABLED": "1"}, clear=False):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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

        with patch.dict(os.environ, {"AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "dry-run"}):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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

    def test_openai_canary_applies_only_selected_safe_openai_model(self):
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
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-mini")
        [row] = server.store.conn.execute("select routed_model, routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        managed = routing["managed_recommendation"]

        self.assertEqual(row["routed_model"], "gpt-5-mini")
        self.assertEqual(managed["mode"], "canary")
        self.assertEqual(managed["status"], "applied")
        self.assertTrue(managed["applied"])
        self.assertTrue(managed["changed_model"])
        self.assertEqual(managed["canary"]["cohort"], "canary_applied")
        self.assertEqual(routing["final_policy_source"], "managed-recommended")

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
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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
            "schema": "agentflow.policy_decision.v1",
            "enabled": True,
            "status": "received",
            "provider": "openai",
            "source_surface": "openai_chat",
            "confidence": 0.94,
            "policy_id": "managed-local-actions",
            "reason": "feature-only policy decision",
            "routing": {"target_model": "gpt-5-mini"},
            "crunch": {"profile": "aggressive", "threshold_chars": 1000},
            "cache": {"profile": "semantic", "semantic_threshold": 0.82},
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }

        with patch.dict(os.environ, {
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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

    def test_openai_policy_decision_rejects_raw_like_unsafe_actions(self):
        request_body = {"model": "gpt-5-codex", "input": "short prompt"}
        recommendation = {
            "schema": "agentflow.policy_decision.v1",
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
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }):
            with patch(
                "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
                return_value=recommendation,
            ):
                with patch.object(openai_proxy.httpx, "AsyncClient", CapturingOpenAIClient):
                    response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CapturingOpenAIClient.calls[0]["json"]["model"], "gpt-5-codex")
        [row] = server.store.conn.execute("select routing_json, crunch_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        crunch = json.loads(row["crunch_json"])
        self.assertEqual(routing["managed_recommendation"]["status"], "skipped")
        self.assertEqual(routing["managed_recommendation"]["apply_reason"], "privacy-not-metadata-only")
        self.assertEqual(crunch["policy_source"], "local-default")

    def test_openai_policy_decision_rejects_expired_provider_mismatch_and_unknown_actions(self):
        cases = [
            (
                {
                    "schema": "agentflow.policy_decision.v1",
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
                    "schema": "agentflow.policy_decision.v1",
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
                    "schema": "agentflow.policy_decision.v1",
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
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
                    "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
                }):
                    with patch(
                        "agentflow_proxy.optimization.openai_recommendations.fetch_recommendation",
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
                    self.assertEqual(CapturingOpenAIClient.calls[-1]["json"]["model"], "gpt-5-mini")
                    self.assertEqual(
                        managed["local_actions"]["unsupported_actions"][0]["reason"],
                        "unsupported-action-type",
                    )
                    self.assertEqual(managed["status"], "applied")
                else:
                    self.assertEqual(CapturingOpenAIClient.calls[-1]["json"]["model"], "gpt-5-codex")
                    self.assertEqual(managed["status"], "skipped")
                    self.assertEqual(managed["apply_reason"], expected)

    def test_openai_policy_decision_falls_back_when_managed_disabled(self):
        with patch.dict(os.environ, {
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "canary",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION": "1",
        }, clear=False):
            os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
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


if __name__ == "__main__":
    unittest.main()
