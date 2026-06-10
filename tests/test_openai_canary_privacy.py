from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from agentflow_proxy import cli
from agentflow_proxy import router as router_module
from agentflow_proxy.openai_canary_impact import build_openai_canary_impact_report
from agentflow_proxy.openai_routing_report import build_openai_routing_report
from agentflow_proxy.optimization.openai_outcomes import record_managed_outcome_feedback
from agentflow_proxy.stats import stats_openai_canary_readiness
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


FORBIDDEN_VALUES = (
    "raw-canary-prompt-secret",
    "raw-canary-message-secret",
    "raw-canary-content-secret",
    "raw-canary-tool-payload-secret",
    "req_canary_raw_secret",
    "cache-key-canary-secret",
    "/home/lutz/private/canary_secret.py",
    "sk-canary-secret",
    "tenant-canary-secret",
    "raw-openai-session-canary",
)

FORBIDDEN_KEYS = (
    '"api_key"',
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages"',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"session_id"',
    '"tenant_id"',
    '"tool_payload"',
)


def _assert_openai_canary_privacy_clean(testcase: unittest.TestCase, payload: object) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_VALUES:
        testcase.assertNotIn(forbidden, rendered)
    for forbidden_key in FORBIDDEN_KEYS:
        testcase.assertNotIn(forbidden_key, rendered)


def _raw_like_request_json() -> str:
    return stable_json(
        {
            "model": "gpt-5-codex",
            "input": [
                {"role": "user", "content": "raw-canary-prompt-secret"},
                {"type": "message", "content": "raw-canary-message-secret"},
                {"type": "function_call", "arguments": "raw-canary-tool-payload-secret"},
            ],
            "messages": [{"role": "user", "content": "raw-canary-content-secret"}],
            "metadata": {
                "request_id": "req_canary_raw_secret",
                "session_id": "raw-openai-session-canary",
                "cache_key": "cache-key-canary-secret",
                "tenant_id": "tenant-canary-secret",
                "api_key": "sk-canary-secret",
                "file_path": "/home/lutz/private/canary_secret.py",
            },
            "tools": [{"type": "function", "name": "lookup", "description": "raw-canary-tool-payload-secret"}],
        }
    )


def _raw_like_extra_fields() -> dict[str, object]:
    return {
        "prompt": "raw-canary-prompt-secret",
        "messages": [{"role": "user", "content": "raw-canary-message-secret"}],
        "content": "raw-canary-content-secret",
        "tool_payload": {"arguments": "raw-canary-tool-payload-secret"},
        "request_id": "req_canary_raw_secret",
        "cache_key": "cache-key-canary-secret",
        "file_path": "/home/lutz/private/canary_secret.py",
        "session_id": "raw-openai-session-canary",
        "tenant_id": "tenant-canary-secret",
        "api_key": "sk-canary-secret",
        "raw_request": json.loads(_raw_like_request_json()),
    }


class FakeQueuedFeedbackStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.updated: list[tuple[str, dict[str, object]]] = []

    def enqueue_managed_outcome_feedback(self, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))

    def get_managed_outcome_feedback(self, queue_id: str) -> dict[str, object] | None:
        for row in self.rows:
            if row.get("id") == queue_id:
                return row
        return None

    def update_call_routing_json(self, call_id: str, routing_json: str) -> None:
        self.updated.append((call_id, json.loads(routing_json)))


class OpenAICanaryPrivacyFixturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)
        self.saved_routing_rules = os.environ.get("AGENTFLOW_ROUTING_RULES")

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()
        if self.saved_routing_rules is None:
            os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
        else:
            os.environ["AGENTFLOW_ROUTING_RULES"] = self.saved_routing_rules
        importlib.reload(router_module)

    def _log_openai_call(
        self,
        *,
        call_id: str | None = None,
        cohort: str | None = None,
        status_code: int = 200,
        category: str = "chat",
        text_chars: int = 1200,
    ) -> str:
        call_id = call_id or str(uuid.uuid4())
        actual_input_tokens = max(1, text_chars // 4)
        routed_model = "gpt-5-mini" if cohort == "canary_applied" else "gpt-5-codex"
        canary = None
        if cohort:
            status = "applied" if cohort == "canary_applied" else "holdout"
            canary = {
                "enabled": True,
                "policy_id": "local-openai-canary-privacy",
                "rule_id": "local-openai-canary-privacy",
                "promotion_action_id": "openai-canary-action-privacy",
                "target_candidate_id": "openai-canary-candidate-privacy",
                "candidate_id": "openai-canary-candidate-privacy",
                "status": status,
                "cohort": cohort,
                "reason": "selected-canary" if status == "applied" else "selected-holdout",
                "original_model": "gpt-5-codex",
                "requested_model": "gpt-5-codex",
                "target_model": "gpt-5-mini",
                "actual_forwarded_model": routed_model,
                "source_surface": "openai_provider_request",
                "app_family": "generic_openai",
                "category": category,
                "projected_input_savings_usd": 0.002,
                "canary_fraction": 0.5,
                "holdout_fraction": 0.25,
                "policy_source": "local-manual",
                "cohort_key_hash": f"sha256:{call_id}",
                **_raw_like_extra_fields(),
            }
        routing = {
            "enabled": bool(cohort),
            "provider": "openai",
            "requested_model": "gpt-5-codex",
            "routed_model": routed_model,
            "reason": "test-openai-canary-privacy",
            "text_chars": text_chars,
            "has_tools": False,
            "category": category,
            "policy_source": "local-default",
            **_raw_like_extra_fields(),
        }
        if canary is not None:
            routing["openai_canary"] = canary
        self.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model=routed_model,
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=125,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=40,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=40,
            cost_est_usd=0.001 if cohort == "canary_applied" else 0.003,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0, **_raw_like_extra_fields()}),
            routing_json=stable_json(routing),
            cache_json=stable_json(
                {
                    "status": "miss",
                    "reason": "exact-miss",
                    "policy_source": "local-default",
                    **_raw_like_extra_fields(),
                }
            ),
            error=stable_json({"error": {"type": "upstream_error", "message": "raw-canary-content-secret"}})
            if status_code >= 400
            else None,
            request_json=_raw_like_request_json(),
            response_json=stable_json({"output_text": "raw-canary-content-secret"}),
            session_id="raw-openai-session-canary",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )
        return call_id

    def test_openai_routing_report_ignores_raw_request_and_decision_fields(self) -> None:
        for _ in range(5):
            self._log_openai_call(category="chat", text_chars=1200)

        report = build_openai_routing_report(self.store, limit=10)

        self.assertEqual(report["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(report["summary"]["openai_call_count"], 5)
        self.assertEqual(report["summary"]["candidate_count"], 1)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "openai-route:responses:gpt-5:chat:no-tools:nonstream:lt-1_5k:lt-1k:to-gpt-5-mini")
        self.assertEqual(candidate["target_model"], "gpt-5-mini")
        self.assertEqual(candidate["matched_count"], 5)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        _assert_openai_canary_privacy_clean(self, report)

    def test_openai_canary_policy_loader_and_route_metadata_are_metadata_only(self) -> None:
        policy_path = Path(self.tmpdir.name) / "canary-rules.yaml"
        policy_path.write_text(
            "\n".join(
                [
                    "openai_canary:",
                    "  enabled: true",
                    "  policy_id: local-openai-canary-privacy",
                    "  promotion_action_id: openai-canary-action-privacy",
                    "  target_candidate_id: openai-canary-candidate-privacy",
                    "  model_pattern: gpt-5",
                    "  target_model: gpt-5-mini",
                    "  eligible_categories: [short-completion]",
                    "  excluded_categories: []",
                    "  allow_tools: false",
                    "  allow_stream: false",
                    "  min_text_chars: 0",
                    "  max_text_chars: 8000",
                    "  min_input_tokens_est: 0",
                    "  max_input_tokens_est: 2000",
                    "  canary_fraction: 1.0",
                    "  holdout_fraction: 0.0",
                    "  salt: openai-canary-privacy-salt",
                    "  safety_stop:",
                    "    enabled: false",
                    "  request_id: req_canary_raw_secret",
                    "  cache_key: cache-key-canary-secret",
                    "  file_path: /home/lutz/private/canary_secret.py",
                    "  tenant_id: tenant-canary-secret",
                    "  api_key: sk-canary-secret",
                    "rules: []",
                    "",
                ]
            )
        )
        os.environ["AGENTFLOW_ROUTING_RULES"] = str(policy_path)
        router = importlib.reload(router_module)

        routed, meta = router.route_openai_model(
            {
                "model": "gpt-5-codex",
                "input": "raw-canary-prompt-secret",
                "metadata": {
                    "request_id": "req_canary_raw_secret",
                    "session_id": "raw-openai-session-canary",
                    "tenant_id": "tenant-canary-secret",
                },
            }
        )

        self.assertEqual(routed, "gpt-5-mini")
        self.assertEqual(meta["openai_canary"]["status"], "applied")
        self.assertEqual(meta["openai_canary"]["candidate_id"], "openai-canary-candidate-privacy")
        self.assertEqual(meta["openai_canary"]["cohort"], "canary_applied")
        self.assertNotIn("request_id", router.ROUTING_OPENAI_CANARY)
        self.assertNotIn("cache_key", router.ROUTING_OPENAI_CANARY)
        self.assertNotIn("file_path", router.ROUTING_OPENAI_CANARY)
        self.assertNotIn("tenant_id", router.ROUTING_OPENAI_CANARY)
        self.assertNotIn("api_key", router.ROUTING_OPENAI_CANARY)
        _assert_openai_canary_privacy_clean(self, router.ROUTING_OPENAI_CANARY)
        _assert_openai_canary_privacy_clean(self, meta)

    def test_openai_canary_impact_cli_and_dashboard_readiness_are_metadata_only(self) -> None:
        self._log_openai_call(call_id="privacy-canary-a1", cohort="canary_applied")
        self._log_openai_call(call_id="privacy-canary-a2", cohort="canary_applied")
        self._log_openai_call(call_id="privacy-canary-h1", cohort="canary_holdout")
        self._log_openai_call(call_id="privacy-canary-e1", cohort="canary_applied", status_code=500)

        impact = build_openai_canary_impact_report(self.store, limit=10)
        self.assertEqual(impact["schema"], "agentflow.openai_canary_impact.v1")
        self.assertEqual(impact["summary"]["observed_openai_canary_metadata_row_count"], 4)
        self.assertEqual(impact["candidates"][0]["candidate_id"], "openai-canary-candidate-privacy")
        self.assertEqual(impact["candidates"][0]["cohort_counts"]["canary_applied"], 3)
        self.assertFalse(impact["privacy"]["raw_prompts_included"])
        self.assertFalse(impact["privacy"]["request_ids_included"])
        self.assertFalse(impact["privacy"]["cache_keys_included"])
        _assert_openai_canary_privacy_clean(self, impact)

        cli_output = io.StringIO()
        exit_code = cli.openai_canary_impact_cli(["--db", self.db_path, "--limit", "10"], stdout=cli_output)
        self.assertEqual(exit_code, 0)
        cli_payload = json.loads(cli_output.getvalue())
        self.assertEqual(cli_payload["schema"], "agentflow.openai_canary_impact.v1")
        _assert_openai_canary_privacy_clean(self, cli_payload)

        with patch.dict(
            router_module.ROUTING_OPENAI_CANARY,
            {
                "enabled": True,
                "policy_id": "local-openai-canary-privacy",
                "target_candidate_id": "openai-canary-candidate-privacy",
                "target_model": "gpt-5-mini",
                "canary_fraction": 0.5,
                "holdout_fraction": 0.25,
                "policy_source": "local-manual",
            },
            clear=False,
        ):
            with patch.object(router_module, "ROUTING_RULES_PATH", "/home/lutz/private/canary_secret.py"):
                readiness = asyncio.run(stats_openai_canary_readiness(self.store, limit=10))

        self.assertEqual(readiness["schema"], "agentflow.openai_canary_readiness.v1")
        self.assertTrue(readiness["read_only"])
        self.assertFalse(readiness["provider_calls_made"])
        self.assertFalse(readiness["managed_server_calls_made"])
        self.assertFalse(readiness["privacy"]["policy_file_paths_included"])
        self.assertFalse(readiness["privacy"]["tenant_ids_included"])
        self.assertNotIn("rule_path", readiness["policy"])
        self.assertEqual(readiness["policy"]["rule_path_included"], False)
        self.assertEqual(readiness["summary"]["canary_holdout_count"], 1)
        _assert_openai_canary_privacy_clean(self, readiness)

    def test_openai_managed_outcome_feedback_queue_payload_is_metadata_only(self) -> None:
        store = FakeQueuedFeedbackStore()
        routing_meta = {
            "reason": "OpenAI canary selected local route",
            "policy_source": "local-manual",
            "managed_policy_id": "local-openai-canary-privacy",
            "managed_reason": "selected-canary",
            "category": "chat",
            "managed_recommendation": {
                "enabled": True,
                "optimization_unit_id": 77,
                "recommendation_id": "rec-openai-canary-privacy",
                "policy_id": "local-openai-canary-privacy",
                "target_model": "gpt-5-mini",
                "mode": "canary",
                "status": "applied",
                "lifecycle_event": "canary_applied",
                "applied": True,
                "changed_model": True,
                "apply_reason": "canary-selected",
                "canary": {"enabled": True, "cohort": "canary_applied"},
                **_raw_like_extra_fields(),
            },
            "openai_canary": {
                "enabled": True,
                "policy_id": "local-openai-canary-privacy",
                "candidate_id": "openai-canary-candidate-privacy",
                "status": "applied",
                "cohort": "canary_applied",
                "reason": "selected-canary",
                **_raw_like_extra_fields(),
            },
            **_raw_like_extra_fields(),
        }

        with patch.dict(os.environ, {"AGENTFLOW_RECOMMENDATION_ENABLED": "1"}):
            asyncio.run(
                record_managed_outcome_feedback(
                    store=store,
                    call_id="call-openai-canary-privacy",
                    path="/v1/responses",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-mini",
                    status_code=200,
                    latency_ms=125,
                    retry_count=0,
                    input_tokens_est=300,
                    output_tokens_est=40,
                    actual_input_tokens=300,
                    actual_output_tokens=40,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=0,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.003,
                    cache_meta={"status": "miss", "reason": "exact-miss", **_raw_like_extra_fields()},
                    crunch_meta={"changed": False, **_raw_like_extra_fields()},
                    routing_meta=routing_meta,
                    category="chat",
                    session_id="raw-openai-session-canary",
                    error='{"error":{"type":"upstream_error","message":"raw-canary-content-secret"}}',
                )
            )

        self.assertEqual(len(store.rows), 1)
        payload = json.loads(str(store.rows[0]["payload_json"]))
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["source_surface"], "openai_responses")
        self.assertEqual(payload["managed_recommendation"]["optimization_unit_id"], 77)
        self.assertEqual(payload["managed_recommendation"]["canary_cohort"], "canary_applied")
        self.assertEqual(payload["session"]["present"], True)
        self.assertTrue(payload["session"]["id_hash"])
        self.assertFalse(payload["raw_payload_included"] if "raw_payload_included" in payload else False)
        _assert_openai_canary_privacy_clean(self, payload)
        self.assertEqual(len(store.updated), 1)
        public_feedback_meta = store.updated[0][1]["managed_recommendation"]["outcome_feedback"]
        self.assertEqual(public_feedback_meta["status"], "queued")
        _assert_openai_canary_privacy_clean(self, public_feedback_meta)


if __name__ == "__main__":
    unittest.main()
