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
        self.assertTrue(feature_summary["has_tools"])
        self.assertFalse(feature_summary["raw_payload_included"])
        self.assertEqual(outcome_summary["status_code"], 200)
        self.assertEqual(outcome_summary["actual_input_tokens"], 11)
        self.assertEqual(outcome_summary["actual_output_tokens"], 3)
        self.assertEqual(outcome_summary["source_surface"], "openai_responses")
        self.assertFalse(outcome_summary["raw_payload_included"])
        self.assertNotIn("raw openai prompt", row["routing_json"])
        self.assertNotIn("raw tool output", row["routing_json"])

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


if __name__ == "__main__":
    unittest.main()
