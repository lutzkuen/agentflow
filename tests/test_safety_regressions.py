import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import routing_experiments
    from agentflow_proxy import openai_proxy, server
    from agentflow_proxy.limiter import TierBackoffActive
    from agentflow_proxy.store import Store


class FakeJsonResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = json.dumps(body).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self):
        return self._body


class CapturingAsyncClient:
    calls = []
    response_body = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        CapturingAsyncClient.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        return FakeJsonResponse(CapturingAsyncClient.response_body)


class ManagedFeedbackAsyncClient:
    calls = []
    provider_body = {}
    provider_status = 200
    recommendation_body = None
    recommendation_status = 200
    recommendation_error = None
    feedback_error = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        if url.endswith("/v1/recommendation"):
            if ManagedFeedbackAsyncClient.recommendation_error is not None:
                raise ManagedFeedbackAsyncClient.recommendation_error
            ManagedFeedbackAsyncClient.calls.append({
                "kind": "recommendation",
                "url": url,
                "headers": dict(headers or {}),
                "json": json,
                "kwargs": kwargs,
            })
            if ManagedFeedbackAsyncClient.recommendation_body is not None:
                return FakeJsonResponse(
                    ManagedFeedbackAsyncClient.recommendation_body,
                    ManagedFeedbackAsyncClient.recommendation_status,
                )
            requested = str((json or {}).get("requested_model") or "")
            target = "claude-haiku-4-5-20251001" if requested.startswith("claude-") else requested
            return FakeJsonResponse({
                "target_model": target,
                "replacement_prompt": None,
                "confidence": 0.91,
                "policy_id": "policy-managed-test",
                "reason": "test recommendation",
                "optimization_unit_id": 42,
            })
        ManagedFeedbackAsyncClient.calls.append({
            "kind": "upstream",
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        return FakeJsonResponse(ManagedFeedbackAsyncClient.provider_body, ManagedFeedbackAsyncClient.provider_status)

    async def patch(self, url, *, json=None, **kwargs):
        ManagedFeedbackAsyncClient.calls.append({
            "kind": "feedback",
            "url": url,
            "json": json,
            "kwargs": kwargs,
        })
        if ManagedFeedbackAsyncClient.feedback_error is not None:
            raise ManagedFeedbackAsyncClient.feedback_error
        return FakeJsonResponse({"ok": True})


class SequencedAsyncClient:
    calls = []
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        SequencedAsyncClient.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        response = SequencedAsyncClient.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingSemaphore:
    def __init__(self, limiter, tier):
        self.limiter = limiter
        self.tier = tier

    async def __aenter__(self):
        self.limiter.entered.append(self.tier)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.limiter.exited.append(self.tier)
        return False


class RecordingLimiter:
    def __init__(self, *, raise_backoff=False):
        self.raise_backoff = raise_backoff
        self.awaited = []
        self.throttled = 0
        self.recorded_backoffs = []
        self.entered = []
        self.exited = []
        self.semaphores = {
            "haiku": RecordingSemaphore(self, "haiku"),
            "sonnet": RecordingSemaphore(self, "sonnet"),
            "opus": RecordingSemaphore(self, "opus"),
        }

    async def await_backoff(self, model):
        self.awaited.append(model)
        if self.raise_backoff:
            raise TierBackoffActive(tier="sonnet", remaining=45.0)

    async def throttle_forward(self):
        self.throttled += 1

    async def record_backoff(self, model, response_headers, default_seconds=60.0):
        self.recorded_backoffs.append({
            "model": model,
            "retry_after": response_headers.get("retry-after"),
            "default_seconds": default_seconds,
        })


async def noop_sleep(_delay):
    return None


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class SafetyRegressionRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_limiter = server._limiter
        self.old_tier_backoff_until = server._tier_backoff_until
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.old_log_bodies = server.LOG_BODIES
        self.old_routing_experiment_enabled = routing_experiments.ROUTING_EXPERIMENT_ENABLED
        self.recommendation_env_keys = (
            "AGENTFLOW_RECOMMENDATION_ENABLED",
            "AGENTFLOW_RECOMMENDATION_SERVER_URL",
            "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS",
            "AGENTFLOW_OPENAI_RECOMMENDATION_MODE",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION",
            "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_SALT",
        )
        self.saved_recommendation_env = {key: os.environ.get(key) for key in self.recommendation_env_keys}
        for key in self.recommendation_env_keys:
            os.environ.pop(key, None)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)
        server.LOG_BODIES = False
        routing_experiments.ROUTING_EXPERIMENT_ENABLED = False
        CapturingAsyncClient.calls = []
        ManagedFeedbackAsyncClient.calls = []
        ManagedFeedbackAsyncClient.provider_body = {}
        ManagedFeedbackAsyncClient.provider_status = 200
        ManagedFeedbackAsyncClient.recommendation_body = None
        ManagedFeedbackAsyncClient.recommendation_status = 200
        ManagedFeedbackAsyncClient.recommendation_error = None
        ManagedFeedbackAsyncClient.feedback_error = None
        SequencedAsyncClient.calls = []
        SequencedAsyncClient.responses = []

    def tearDown(self):
        for key, value in self.saved_recommendation_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server._limiter = self.old_limiter
        server._tier_backoff_until = self.old_tier_backoff_until
        server.LOG_BODIES = self.old_log_bodies
        routing_experiments.ROUTING_EXPERIMENT_ENABLED = self.old_routing_experiment_enabled
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_anthropic_upstream,
            openai_upstream=self.old_openai_upstream,
            openai_auth_mode=self.old_openai_auth_mode,
        )

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

    def _managed_feedback_env(self):
        return patch.dict(os.environ, {
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
            "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
            "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS": "0.25",
        }, clear=False)

    def test_log_bodies_defaults_disabled_when_env_is_absent(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as db:
            env = os.environ.copy()
            env.pop("AGENTFLOW_LOG_BODIES", None)
            env["AGENTFLOW_DB"] = db.name
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from agentflow_proxy import server; print(int(server.LOG_BODIES))",
                ],
                cwd=os.getcwd(),
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertEqual(result.stdout.strip(), "0")

    def test_anthropic_managed_recommendation_sends_sanitized_outcome_feedback(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "raw prompt secret"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual([call["kind"] for call in ManagedFeedbackAsyncClient.calls], ["recommendation", "upstream", "feedback"])
        recommendation = ManagedFeedbackAsyncClient.calls[0]["json"]
        feedback = ManagedFeedbackAsyncClient.calls[2]["json"]
        self.assertEqual(ManagedFeedbackAsyncClient.calls[2]["url"], "http://managed.test/v1/optimization-units/42/outcome")
        self.assertEqual(feedback["status_code"], 200)
        self.assertEqual(feedback["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(feedback["managed_recommendation"]["optimization_unit_id"], 42)
        self.assertEqual(feedback["quality_signals"]["status"], "success")
        self.assertIn("success", feedback["quality_signals"]["signal_codes"])
        self.assertTrue({"messages", "content", "raw_request"}.isdisjoint(self._keys_in(recommendation)))
        self.assertTrue({"messages", "content", "raw_response"}.isdisjoint(self._keys_in(feedback)))
        self.assertNotIn("raw prompt secret", str(recommendation))
        self.assertNotIn("raw prompt secret", str(feedback))
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        routing = json.loads(row["routing_json"])
        feedback_meta = routing["managed_recommendation"]["outcome_feedback"]
        self.assertEqual(feedback_meta["status"], "sent")
        self.assertEqual(feedback_meta["optimization_unit_id"], 42)
        [queue_row] = server.store.conn.execute(
            "select source_surface, status, attempts, payload_json from managed_outcome_feedback_queue"
        ).fetchall()
        self.assertEqual(queue_row["source_surface"], "anthropic_messages")
        self.assertEqual(queue_row["status"], "sent")
        self.assertEqual(queue_row["attempts"], 1)
        self.assertNotIn("raw prompt secret", queue_row["payload_json"])

    def test_anthropic_managed_recommendation_records_feature_unit_and_model_change(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.recommendation_body = {
            "target_model": "claude-sonnet-4-6",
            "replacement_prompt": None,
            "confidence": 0.91,
            "policy_id": "policy-managed-test",
            "reason": "test recommendation",
            "optimization_unit_id": 42,
        }
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "raw prompt secret"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post(
                "/v1/messages",
                json=request_body,
                headers={"x-session-id": "session-managed-secret"},
            )

        self.assertEqual(response.status_code, 200)
        recommendation = ManagedFeedbackAsyncClient.calls[0]["json"]
        upstream = ManagedFeedbackAsyncClient.calls[1]["json"]
        self.assertEqual(recommendation["feature_schema_version"], "agentflow.optimization_unit_features.v1")
        self.assertEqual(recommendation["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertTrue(recommendation["privacy_summary"]["metadata_only"])
        self.assertFalse(recommendation["privacy_summary"]["raw_body_storage"])
        self.assertIn("session_id_hash", recommendation["grouping_identifiers"])
        self.assertTrue(recommendation["grouping_identifiers"]["session_id_hash"].startswith("sha256:"))
        self.assertNotIn("session-managed-secret", str(recommendation))
        self.assertTrue({"messages", "content", "raw_request"}.isdisjoint(self._keys_in(recommendation)))
        self.assertNotIn("raw prompt secret", str(recommendation))
        self.assertEqual(upstream["model"], "claude-sonnet-4-6")
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        managed = json.loads(row["routing_json"])["managed_recommendation"]
        self.assertEqual(managed["status"], "received")
        self.assertTrue(managed["applied"])
        self.assertTrue(managed["changed_model"])

    def test_anthropic_managed_recommendation_rejects_replacement_prompt(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.recommendation_body = {
            "target_model": "claude-sonnet-4-6",
            "replacement_prompt": "raw replacement prompt secret",
            "confidence": 0.91,
            "policy_id": "policy-replacement-disabled",
            "reason": "replace unsafe prompt",
            "optimization_unit_id": 43,
        }
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "raw prompt secret"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        upstream = ManagedFeedbackAsyncClient.calls[1]["json"]
        self.assertEqual(upstream["model"], "claude-haiku-4-5-20251001")
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        managed = json.loads(row["routing_json"])["managed_recommendation"]
        self.assertEqual(managed["status"], "received")
        self.assertFalse(managed["applied"])
        self.assertEqual(managed["apply_reason"], "unsafe-replacement-prompt")
        self.assertFalse(managed["replacement_prompt_applied"])
        self.assertIn("replacement_prompt_sha256", managed)
        self.assertNotIn("raw replacement prompt secret", str(managed))

    def test_anthropic_managed_server_error_falls_back_to_provider_call(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.recommendation_body = {"error": "managed unavailable"}
        ManagedFeedbackAsyncClient.recommendation_status = 503
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "raw prompt secret"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual([call["kind"] for call in ManagedFeedbackAsyncClient.calls], ["recommendation", "upstream"])
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        managed = json.loads(row["routing_json"])["managed_recommendation"]
        self.assertEqual(managed["status"], "error")
        self.assertEqual(managed["reason"], "server-error")
        self.assertEqual(managed["fallback"], "local-policy")
        self.assertFalse(managed["applied"])

    def test_anthropic_routing_experiment_exports_metadata_only_feedback(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "primary raw output secret"}],
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "short prompt secret"}],
        }

        patches = [
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_ENABLED", True),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_MODE", "applied_routed_down"),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_SAMPLE_RATE", 1.0),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_DAILY_BUDGET_USD", 0.05),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD", 0.86),
            patch.dict(routing_experiments.ROUTING_EXPERIMENT_POLICY, {
                "mode": "applied_routed_down",
                "providers": ["anthropic"],
                "source_surfaces": ["anthropic_messages"],
                "model_pairs": [
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                    }
                ],
                "categories": [],
                "min_text_chars": 0,
                "max_text_chars": 30000,
                "daily_budget_usd": 0.05,
            }),
        ]
        with (
            self._managed_feedback_env(),
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient),
        ):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual([call["kind"] for call in ManagedFeedbackAsyncClient.calls], ["recommendation", "upstream", "upstream", "feedback"])
        feedback = ManagedFeedbackAsyncClient.calls[-1]["json"]
        experiment = feedback["routing_experiment"]
        self.assertEqual(experiment["schema"], "agentflow.routing_experiment_feedback.v1")
        self.assertEqual(experiment["primary_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(experiment["shadow_model"], "claude-sonnet-4-6")
        self.assertEqual(experiment["output_similarity"], 1.0)
        self.assertIn("primary_output_sha256", experiment)
        self.assertNotIn("primary raw output secret", str(feedback))
        self.assertNotIn("short prompt secret", str(feedback))

        [call_row] = server.store.conn.execute("select routing_json from calls").fetchall()
        routing = json.loads(call_row["routing_json"])
        self.assertEqual(routing["routing_experiment"]["managed_feedback"]["status"], "sent")
        [experiment_row] = server.store.conn.execute("select experiment_json from routing_experiments").fetchall()
        experiment_json = json.loads(experiment_row["experiment_json"])
        self.assertEqual(experiment_json["managed_feedback"]["status"], "sent")
        self.assertEqual(experiment_json["optimization_feedback"]["output_similarity"], 1.0)

    def test_anthropic_shadow_experiment_keeps_primary_requested_model(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        SequencedAsyncClient.responses = [
            FakeJsonResponse({
                "id": "msg_primary",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "primary safe output"}],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            }),
            FakeJsonResponse({
                "id": "msg_shadow",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": "primary safe output"}],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            }),
        ]
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "short prompt secret"}],
        }

        patches = [
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_ENABLED", True),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_MODE", "shadow_candidate_pass_through"),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_SAMPLE_RATE", 1.0),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_DAILY_BUDGET_USD", 10.0),
            patch.dict(routing_experiments.ROUTING_EXPERIMENT_POLICY, {
                "mode": "shadow_candidate_pass_through",
                "providers": ["anthropic"],
                "source_surfaces": ["anthropic_messages"],
                "model_pairs": [
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                    }
                ],
                "categories": ["short-completion"],
                "min_text_chars": 0,
                "max_text_chars": 30000,
                "daily_budget_usd": 10.0,
            }),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patch.object(server.httpx, "AsyncClient", SequencedAsyncClient),
        ):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "msg_primary")
        self.assertEqual([call["json"]["model"] for call in SequencedAsyncClient.calls], [
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ])
        [call_row] = server.store.conn.execute(
            "select requested_model, routed_model, routing_json from calls"
        ).fetchall()
        self.assertEqual(call_row["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(call_row["routed_model"], "claude-sonnet-4-6")
        routing = json.loads(call_row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertTrue(experiment["sampled"])
        self.assertEqual(experiment["mode"], "shadow_candidate_pass_through")
        self.assertTrue(experiment["counterfactual"])
        self.assertTrue(experiment["shadow_only"])
        self.assertEqual(experiment["provider"], "anthropic")
        self.assertEqual(experiment["source_surface"], "anthropic_messages")
        self.assertEqual(experiment["primary_model"], "claude-sonnet-4-6")
        self.assertEqual(experiment["user_visible_model"], "claude-sonnet-4-6")
        self.assertEqual(experiment["shadow_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(experiment["local_route_candidate_model"], "claude-haiku-4-5-20251001")
        [experiment_row] = server.store.conn.execute(
            "select provider, source_surface, requested_model, routed_model, primary_model, shadow_model, experiment_json "
            "from routing_experiments"
        ).fetchall()
        self.assertEqual(experiment_row["provider"], "anthropic")
        self.assertEqual(experiment_row["source_surface"], "anthropic_messages")
        self.assertEqual(experiment_row["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(experiment_row["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(experiment_row["primary_model"], "claude-sonnet-4-6")
        self.assertEqual(experiment_row["shadow_model"], "claude-haiku-4-5-20251001")
        experiment_json = json.loads(experiment_row["experiment_json"])
        self.assertEqual(experiment_json["optimization_feedback"]["status"], "compared")
        [queue_row] = server.store.conn.execute(
            "select source_surface, status, payload_json from managed_outcome_feedback_queue"
        ).fetchall()
        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        self.assertEqual(queue_row["status"], "queued")
        payload = json.loads(queue_row["payload_json"])
        self.assertEqual(payload["provider"], "anthropic")
        self.assertEqual(payload["source_surface"], "anthropic_messages")
        self.assertTrue(payload["candidate"]["counterfactual"])
        self.assertTrue(payload["candidate"]["shadow_only"])
        self.assertNotIn("short prompt secret", queue_row["payload_json"])
        self.assertNotIn("primary safe output", queue_row["payload_json"])

    def test_anthropic_shadow_failure_is_logged_without_affecting_primary(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        SequencedAsyncClient.responses = [
            FakeJsonResponse({
                "id": "msg_primary",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "primary ok"}],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            }),
            RuntimeError("shadow provider unavailable"),
        ]
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "short prompt secret"}],
        }

        patches = [
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_ENABLED", True),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_MODE", "shadow_candidate_pass_through"),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_SAMPLE_RATE", 1.0),
            patch.object(routing_experiments, "ROUTING_EXPERIMENT_DAILY_BUDGET_USD", 10.0),
            patch.dict(routing_experiments.ROUTING_EXPERIMENT_POLICY, {
                "mode": "shadow_candidate_pass_through",
                "providers": ["anthropic"],
                "source_surfaces": ["anthropic_messages"],
                "model_pairs": [
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                    }
                ],
                "categories": ["short-completion"],
                "min_text_chars": 0,
                "max_text_chars": 30000,
                "daily_budget_usd": 10.0,
            }),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patch.object(server.httpx, "AsyncClient", SequencedAsyncClient),
        ):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "msg_primary")
        self.assertEqual([call["json"]["model"] for call in SequencedAsyncClient.calls], [
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ])
        [experiment_row] = server.store.conn.execute("select error, experiment_json from routing_experiments").fetchall()
        self.assertIn("shadow provider unavailable", experiment_row["error"])
        experiment_json = json.loads(experiment_row["experiment_json"])
        self.assertEqual(experiment_json["optimization_feedback"]["status"], "shadow-error")
        self.assertIn("shadow-exception", experiment_json["optimization_feedback"]["reason_codes"])
        [queue_row] = server.store.conn.execute(
            "select source_surface, status, payload_json from managed_outcome_feedback_queue"
        ).fetchall()
        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        payload = json.loads(queue_row["payload_json"])
        self.assertEqual(payload["outcome"]["status"], "shadow-error")
        self.assertTrue(payload["outcome"]["error_present"])
        self.assertNotIn("short prompt secret", queue_row["payload_json"])
        self.assertNotIn("primary ok", queue_row["payload_json"])

    def test_anthropic_provider_failure_still_returns_and_sends_feedback(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.provider_status = 400
        ManagedFeedbackAsyncClient.provider_body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "provider rejected request"},
        }
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "raw failing prompt"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 400)
        self.assertEqual([call["kind"] for call in ManagedFeedbackAsyncClient.calls], ["recommendation", "upstream", "feedback"])
        feedback = ManagedFeedbackAsyncClient.calls[2]["json"]
        self.assertEqual(feedback["status_code"], 400)
        self.assertEqual(feedback["error_class"], "invalid_request_error")
        self.assertEqual(feedback["quality_signals"]["status"], "failure")
        self.assertIn("failure", feedback["quality_signals"]["signal_codes"])
        self.assertTrue(feedback["error_present"])
        self.assertFalse(feedback["raw_error_included"])
        self.assertNotIn("provider rejected request", str(feedback))
        self.assertNotIn("raw failing prompt", str(feedback))

    def test_anthropic_feedback_failure_is_silent_and_recorded_locally(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        ManagedFeedbackAsyncClient.feedback_error = RuntimeError("feedback down")
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }

        with self._managed_feedback_env(), patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        [row] = server.store.conn.execute("select routing_json from calls").fetchall()
        feedback_meta = json.loads(row["routing_json"])["managed_recommendation"]["outcome_feedback"]
        self.assertEqual(feedback_meta["status"], "retryable-error")
        self.assertEqual(feedback_meta["reason"], "request-failed")
        self.assertIn("feedback down", feedback_meta["error"])
        [queue_row] = server.store.conn.execute(
            "select source_surface, status, attempts, last_error, payload_json from managed_outcome_feedback_queue"
        ).fetchall()
        self.assertEqual(queue_row["source_surface"], "anthropic_messages")
        self.assertEqual(queue_row["status"], "retryable-error")
        self.assertEqual(queue_row["attempts"], 1)
        self.assertIn("feedback down", queue_row["last_error"])
        self.assertNotIn("hello", queue_row["payload_json"])

    def test_openai_managed_recommendation_sends_outcome_feedback(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        ManagedFeedbackAsyncClient.provider_body = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }
        request_body = {"model": "gpt-5-codex", "input": "raw openai prompt"}

        with (
            self._managed_feedback_env(),
            patch.dict(os.environ, {"AGENTFLOW_OPENAI_RECOMMENDATION_MODE": "dry-run"}, clear=False),
            patch.object(server.httpx, "AsyncClient", ManagedFeedbackAsyncClient),
        ):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual([call["kind"] for call in ManagedFeedbackAsyncClient.calls], ["recommendation", "upstream", "feedback"])
        feedback = ManagedFeedbackAsyncClient.calls[2]["json"]
        self.assertEqual(feedback["provider"], "openai")
        self.assertEqual(feedback["source_surface"], "openai_responses")
        self.assertEqual(feedback["actual_input_tokens"], 9)
        self.assertEqual(feedback["actual_output_tokens"], 3)
        self.assertNotIn("raw openai prompt", str(feedback))
        rows = server.store.conn.execute(
            "select source_surface, status, attempts, payload_json from managed_outcome_feedback_queue"
        ).fetchall()
        rows_by_surface = {row["source_surface"]: row for row in rows}
        self.assertEqual(set(rows_by_surface), {"openai_optimization_lifecycle", "openai_responses"})
        queue_row = rows_by_surface["openai_responses"]
        self.assertEqual(queue_row["status"], "sent")
        self.assertEqual(queue_row["attempts"], 1)
        self.assertNotIn("raw openai prompt", queue_row["payload_json"])
        lifecycle_row = rows_by_surface["openai_optimization_lifecycle"]
        self.assertEqual(lifecycle_row["status"], "queued")
        lifecycle_payload = json.loads(lifecycle_row["payload_json"])
        self.assertEqual(lifecycle_payload["schema"], "agentflow.openai_optimization_lifecycle_feedback.v1")
        self.assertNotIn("raw openai prompt", lifecycle_row["payload_json"])

    def test_anthropic_route_forwards_allowlisted_headers_and_does_not_log_bodies(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        CapturingAsyncClient.response_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "secret-body-value"}],
        }

        with patch.object(server.httpx, "AsyncClient", CapturingAsyncClient):
            response = TestClient(server.app).post(
                "/v1/messages",
                json=request_body,
                headers={
                    "Authorization": "Bearer client-key",
                    "Anthropic-Beta": "prompt-caching-2024-07-31",
                    "X-Trace-Secret": "local-secret",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingAsyncClient.calls), 1)
        forwarded = {name.lower(): value for name, value in CapturingAsyncClient.calls[0]["headers"].items()}
        self.assertEqual(forwarded["authorization"], "Bearer client-key")
        self.assertEqual(forwarded["anthropic-beta"], "prompt-caching-2024-07-31")
        self.assertEqual(forwarded["content-type"], "application/json")
        self.assertNotIn("x-trace-secret", forwarded)
        self.assertEqual(CapturingAsyncClient.calls[0]["json"]["messages"][0]["content"], "secret-body-value")

        [row] = server.store.conn.execute(
            "select status_code, request_json, response_json from calls"
        ).fetchall()
        self.assertEqual(row["status_code"], 200)
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])

    def test_openai_route_records_tier_backoff_on_retry_after(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter()
        server._limiter = limiter
        SequencedAsyncClient.responses = [
            FakeJsonResponse(
                {"error": {"message": "rate limited", "type": "rate_limit_error"}},
                status_code=429,
                headers={"retry-after": "7", "content-type": "application/json"},
            ),
            FakeJsonResponse(
                {
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-5-codex",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ),
        ]
        request_body = {"model": "gpt-5-codex", "input": "retry after test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient), patch.object(openai_proxy.asyncio, "sleep", noop_sleep):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex", "gpt-5-codex"])
        self.assertEqual(limiter.throttled, 2)
        self.assertEqual(limiter.recorded_backoffs, [{
            "model": "gpt-5-codex",
            "retry_after": "7",
            "default_seconds": 60.0,
        }])
        self.assertEqual(len(SequencedAsyncClient.calls), 2)
        [row] = server.store.conn.execute(
            "select provider, status_code, retry_count from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 200)
        self.assertEqual(row["retry_count"], 1)

    def test_openai_route_returns_local_429_during_long_tier_cooldown(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter(raise_backoff=True)
        server._limiter = limiter
        request_body = {"model": "gpt-5-codex", "input": "cooldown test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "45")
        self.assertEqual(response.headers["x-agentflow-routed-model"], "gpt-5-codex")
        self.assertEqual(response.json()["error"]["type"], "rate_limit_error")
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex"])
        self.assertEqual(SequencedAsyncClient.calls, [])
        [row] = server.store.conn.execute(
            "select provider, status_code, error from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 429)
        self.assertIn("temporarily limiting requests for sonnet tier", row["error"])

    def test_openai_stream_returns_local_rate_limit_event_during_long_tier_cooldown(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter(raise_backoff=True)
        server._limiter = limiter
        request_body = {"model": "gpt-5-codex", "stream": True, "input": "stream cooldown test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient):
            with TestClient(server.app).stream("POST", "/v1/responses", json=request_body) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        self.assertIn("rate_limit_error", body)
        self.assertIn("temporarily limiting requests for sonnet tier", body)
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex"])
        self.assertEqual(SequencedAsyncClient.calls, [])
        [row] = server.store.conn.execute(
            "select provider, status_code, error, stream from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 429)
        self.assertEqual(row["stream"], 1)
        self.assertIn("temporarily limiting requests for sonnet tier", row["error"])

    def test_openai_route_forwards_allowlisted_headers_and_does_not_log_bodies(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        CapturingAsyncClient.response_body = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {"model": "gpt-5-codex", "input": "secret-openai-body"}

        with patch.object(server.httpx, "AsyncClient", CapturingAsyncClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={
                    "Authorization": "Bearer client-key",
                    "OpenAI-Project": "proj_123",
                    "X-Trace-Secret": "local-secret",
                    "Codex-Internal": "local-only",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingAsyncClient.calls), 1)
        forwarded = {name.lower(): value for name, value in CapturingAsyncClient.calls[0]["headers"].items()}
        self.assertEqual(forwarded["authorization"], "Bearer client-key")
        self.assertEqual(forwarded["openai-project"], "proj_123")
        self.assertEqual(forwarded["content-type"], "application/json")
        self.assertNotIn("x-trace-secret", forwarded)
        self.assertNotIn("codex-internal", forwarded)
        self.assertEqual(CapturingAsyncClient.calls[0]["json"]["input"], "secret-openai-body")

        [row] = server.store.conn.execute(
            "select provider, status_code, request_json, response_json from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 200)
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])


if __name__ == "__main__":
    unittest.main()
