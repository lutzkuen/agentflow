"""Tests for the activation-aware savings opportunity report.

Acceptance criteria:
- A fixture with 0 cache hits and unchanged OpenAI/Anthropic routes produces
  an ordered opportunity report naming cache replay blockers and model-routing candidates.
- JSON output includes schema, target, provider, source surface, blocker codes,
  opportunity family, projected savings bucket, privacy summary, and suggested local command.
- The command does not emit raw prompts, provider bodies, request IDs, session IDs,
  or absolute filesystem paths.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.openai_optimization_governor import LIFECYCLE_SOURCE_SURFACE
from agentflow_proxy.savings_report import (
    OPPORTUNITY_FAMILY_ACTIVATION,
    OPPORTUNITY_FAMILY_CACHE_REPLAY,
    OPPORTUNITY_FAMILY_MODEL_ROUTING,
    SAVINGS_BUCKET_UNKNOWN,
    build_savings_report,
)
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


_SECRET_SESSION = "secret-savings-session-id"
_SECRET_PROMPT = "secret raw savings prompt"
_SECRET_REQUEST_ID = "req-savings-secret-id"
_SECRET_PATH = "/home/lutz/private/savings_secret.py"

REQUIRED_OPPORTUNITY_FIELDS = (
    "target",
    "provider",
    "source_surface",
    "opportunity_family",
    "blocker_codes",
    "projected_savings_bucket",
    "suggested_command",
)


def _empty_activation_config() -> dict:
    return {"schema": "agentflow.activation_config.v1", "targets": {}}


def _openai_activation_config() -> dict:
    return {
        "schema": "agentflow.activation_config.v1",
        "targets": {
            "openai": {
                "id": "openai",
                "configured": True,
                "provider": "openai",
                "local_base_url": "http://127.0.0.1:4003/v1",
                "health_url": "http://127.0.0.1:4003/health",
                "upstream_base_url": "https://api.openai.com",
                "openai_auth_mode": "client",
            }
        },
    }


def _both_activation_config() -> dict:
    cfg = _openai_activation_config()
    cfg["targets"]["claude"] = {
        "id": "claude",
        "configured": True,
        "provider": "anthropic",
        "local_base_url": "http://127.0.0.1:4000",
        "health_url": "http://127.0.0.1:4000/health",
        "upstream_base_url": "https://api.anthropic.com",
    }
    return cfg


class _StoreFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_openai_call(
        self,
        *,
        requested_model: str = "gpt-5.4-mini",
        routed_model: str | None = None,
        category: str = "chat",
        text_chars: int = 1200,
        has_tools: bool = False,
        stream: int = 0,
        status_code: int = 200,
        cache_hit: int = 0,
        cache_json_override: dict | None = None,
    ) -> None:
        routed_model = routed_model or requested_model
        tokens = max(1, text_chars // 4)
        cache_json = cache_json_override or {
            "status": "hit" if cache_hit else ("skipped" if has_tools else "miss"),
            "reason": "exact-hit" if cache_hit else ("tools-disabled" if has_tools else "exact-miss"),
            "policy_source": "local-default",
        }
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/chat/completions",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=cache_hit,
            status_code=status_code,
            latency_ms=120,
            input_tokens_est=tokens,
            output_tokens_est=40,
            actual_input_tokens=tokens,
            actual_output_tokens=40,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(
                {
                    "enabled": False,
                    "provider": "openai",
                    "requested_model": requested_model,
                    "routed_model": routed_model,
                    "reason": "openai routing disabled",
                    "text_chars": text_chars,
                    "has_tools": has_tools,
                    "category": category,
                    "policy_source": "local-default",
                }
            ),
            cache_json=stable_json(cache_json),
            error=None,
            request_json=json.dumps({"messages": [{"role": "user", "content": _SECRET_PROMPT}]}),
            response_json=None,
            session_id=_SECRET_SESSION,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_chat",
            endpoint="chat",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )


class TestBuildSavingsReportNoStore(unittest.TestCase):
    def test_unconfigured_targets_produce_activation_opportunities(self) -> None:
        result = build_savings_report(_empty_activation_config())

        self.assertEqual(result["schema"], "agentflow.savings_report.v1")
        self.assertTrue(result["ok"])
        self.assertIn("generated_at", result)
        self.assertIn("privacy", result)
        self.assertEqual(result["opportunity_count"], 2)

        families = {opp["opportunity_family"] for opp in result["opportunities"]}
        self.assertIn(OPPORTUNITY_FAMILY_ACTIVATION, families)

        for opp in result["opportunities"]:
            self.assertIn(opp["opportunity_family"], (OPPORTUNITY_FAMILY_ACTIVATION,))
            self.assertIn("target-not-configured", opp["blocker_codes"])

    def test_privacy_flags_are_set(self) -> None:
        result = build_savings_report(_empty_activation_config())
        privacy = result["privacy"]
        self.assertTrue(privacy["metadata_only"])
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["filesystem_paths_included"])
        self.assertFalse(privacy["provider_calls_made"])
        self.assertFalse(privacy["managed_server_calls_made"])

    def test_all_required_opportunity_fields_present(self) -> None:
        result = build_savings_report(_empty_activation_config())
        for opp in result["opportunities"]:
            for field in REQUIRED_OPPORTUNITY_FIELDS:
                self.assertIn(field, opp, f"Missing field {field!r} in opportunity {opp!r}")

    def test_openai_configured_no_store_produces_no_routing_opportunities(self) -> None:
        result = build_savings_report(_openai_activation_config())
        families = [opp["opportunity_family"] for opp in result["opportunities"]]
        self.assertNotIn(OPPORTUNITY_FAMILY_MODEL_ROUTING, families)
        self.assertNotIn(OPPORTUNITY_FAMILY_CACHE_REPLAY, families)
        # Claude is still unconfigured
        self.assertIn(OPPORTUNITY_FAMILY_ACTIVATION, families)


class TestBuildSavingsReportWithStore(_StoreFixture):
    def test_zero_cache_hits_unchanged_routes_produces_ordered_report(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200, cache_hit=0)
        for _ in range(5):
            self._log_openai_call(category="short-completion", text_chars=700, cache_hit=0)

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)

        self.assertEqual(result["schema"], "agentflow.savings_report.v1")
        self.assertTrue(result["ok"])
        self.assertGreater(result["opportunity_count"], 0)

        families = [opp["opportunity_family"] for opp in result["opportunities"]]
        self.assertIn(OPPORTUNITY_FAMILY_MODEL_ROUTING, families)
        self.assertIn(OPPORTUNITY_FAMILY_CACHE_REPLAY, families)

        # Model-routing should appear before cache-replay in ranked output
        routing_idx = families.index(OPPORTUNITY_FAMILY_MODEL_ROUTING)
        cache_idx = families.index(OPPORTUNITY_FAMILY_CACHE_REPLAY)
        self.assertLessEqual(routing_idx, cache_idx)

    def test_cache_replay_opportunity_names_no_cache_hits_blocker(self) -> None:
        for _ in range(6):
            self._log_openai_call(cache_hit=0)

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)

        cache_opps = [o for o in result["opportunities"] if o["opportunity_family"] == OPPORTUNITY_FAMILY_CACHE_REPLAY]
        self.assertGreater(len(cache_opps), 0)
        cache_opp = cache_opps[0]
        self.assertIn("no-cache-hits", cache_opp["blocker_codes"])
        self.assertEqual(cache_opp["target"], "openai")
        self.assertEqual(cache_opp["provider"], "openai")
        self.assertIsNotNone(cache_opp.get("evidence_window"))
        self.assertEqual(cache_opp["evidence_window"]["cache_hits"], 0)
        self.assertIsNotNone(cache_opp.get("suggested_command"))

    def test_cache_replay_opportunity_includes_zero_hit_blocker_ladder(self) -> None:
        for _ in range(4):
            self._log_openai_call(
                stream=1,
                cache_json_override={
                    "status": "skipped",
                    "reason": "streaming",
                    "policy_source": "local-default",
                    "replayability_level": "features_only",
                },
            )
        for _ in range(2):
            self._log_openai_call(
                has_tools=True,
                cache_json_override={
                    "status": "skipped",
                    "reason": "tools-disabled",
                    "policy_source": "local-default",
                    "replayability_level": "local-exact-response",
                },
            )

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)

        cache_opp = [o for o in result["opportunities"] if o["opportunity_family"] == OPPORTUNITY_FAMILY_CACHE_REPLAY][0]
        self.assertIn("skipped-streaming", cache_opp["blocker_codes"])
        self.assertIn("skipped-tools", cache_opp["blocker_codes"])
        self.assertEqual(cache_opp["cache_blocker_ladder_summary"]["top_blocker_code"], "skipped-streaming")
        self.assertTrue(cache_opp["cache_blocker_ladder_summary"]["bounded_recent_window"])
        self.assertEqual(cache_opp["cache_blocker_ladder"][0]["provider"], "openai")
        self.assertEqual(cache_opp["cache_blocker_ladder"][0]["source_surface"], "openai_chat")
        rendered = json.dumps(cache_opp, sort_keys=True)
        self.assertNotIn(_SECRET_PROMPT, rendered)
        self.assertNotIn(_SECRET_SESSION, rendered)

    def test_routing_opportunity_names_model_routing_candidates(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200)

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)

        routing_opps = [o for o in result["opportunities"] if o["opportunity_family"] == OPPORTUNITY_FAMILY_MODEL_ROUTING]
        self.assertGreater(len(routing_opps), 0)
        routing_opp = routing_opps[0]
        self.assertEqual(routing_opp["target"], "openai")
        self.assertEqual(routing_opp["provider"], "openai")
        self.assertIsNotNone(routing_opp.get("evidence_window"))
        self.assertGreater(routing_opp["evidence_window"]["calls"], 0)
        self.assertIsNotNone(routing_opp.get("suggested_command"))

    def test_no_raw_data_in_json_output(self) -> None:
        for _ in range(5):
            self._log_openai_call()

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)
        rendered = json.dumps(result, sort_keys=True)

        self.assertNotIn(_SECRET_SESSION, rendered)
        self.assertNotIn(_SECRET_PROMPT, rendered)
        self.assertNotIn(_SECRET_REQUEST_ID, rendered)
        self.assertNotIn(_SECRET_PATH, rendered)

    def test_no_absolute_paths_in_json_output(self) -> None:
        for _ in range(5):
            self._log_openai_call()

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)
        rendered = json.dumps(result, sort_keys=True)

        # No absolute paths should appear in the JSON output
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("/root/", rendered)
        self.assertNotIn("/tmp/", rendered)

    def test_all_required_fields_in_each_opportunity(self) -> None:
        for _ in range(6):
            self._log_openai_call()

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)
        for opp in result["opportunities"]:
            for field in REQUIRED_OPPORTUNITY_FIELDS:
                self.assertIn(field, opp, f"Missing field {field!r} in opportunity {opp!r}")

    def test_privacy_flags_with_store(self) -> None:
        for _ in range(5):
            self._log_openai_call()

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)
        privacy = result["privacy"]
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["filesystem_paths_included"])
        self.assertFalse(privacy["provider_calls_made"])
        self.assertFalse(privacy["managed_server_calls_made"])

    def test_savings_report_summarizes_activation_lifecycle_feedback_states(self) -> None:
        now = utc_now()
        for idx, (state, cohort) in enumerate(
            (
                ("holdout_only", "holdout"),
                ("suppressed", "suppressed"),
                ("rollback_required", "rollback_required"),
            )
        ):
            event = {
                "schema": "agentflow.openai_optimization_lifecycle_feedback.v1",
                "event_type": "activation_staged_optimization_lifecycle",
                "occurred_at": now,
                "provider": "openai",
                "source_surface": LIFECYCLE_SOURCE_SURFACE,
                "endpoint": "responses",
                "event_phase": "dry_run",
                "lifecycle_state": state,
                "family_events": [
                    {
                        "action_family": "routing",
                        "cohort": cohort,
                        "status": cohort,
                        "candidate_id": f"candidate-{idx}",
                        "rule_id": f"rule-{idx}",
                        "reason_codes": [state],
                    }
                ],
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_provider_bodies_included": False,
                    "request_ids_included": False,
                    "raw_session_ids_included": False,
                    "cache_keys_included": False,
                    "tenant_ids_included": False,
                    "file_paths_included": False,
                },
            }
            self.store.enqueue_managed_outcome_feedback(
                id=f"activation-lifecycle-{idx}",
                created_at=now,
                updated_at=now,
                source_surface=LIFECYCLE_SOURCE_SURFACE,
                endpoint="/v1/policy-events",
                optimization_unit_id=0,
                payload_json=stable_json(event),
                status="queued",
                attempts=0,
                next_attempt_at=now,
            )

        result = build_savings_report(_openai_activation_config(), store=self.store, limit=50)

        feedback = result["activation_lifecycle_feedback"]
        self.assertEqual(feedback["schema"], "agentflow.activation_staged_lifecycle_feedback_summary.v1")
        self.assertEqual(feedback["queue_rows"], 3)
        states = {item["value"]: item["count"] for item in feedback["state_breakdown"]}
        self.assertEqual(states["holdout_only"], 1)
        self.assertEqual(states["suppressed"], 1)
        self.assertEqual(states["rollback_required"], 1)
        self.assertFalse(feedback["payload_json_included"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(_SECRET_PROMPT, rendered)
        self.assertNotIn(_SECRET_SESSION, rendered)
        self.assertNotIn(_SECRET_REQUEST_ID, rendered)


class TestSavingsReportCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_dir = self.tmpdir.name

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_savings_report_json_no_db_unconfigured(self) -> None:
        stdout = io.StringIO()
        code = cli.agentflow_cli(
            ["savings", "report", "--json", "--config-dir", self.config_dir],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.savings_report.v1")
        self.assertTrue(result["ok"])
        self.assertIn("opportunities", result)
        self.assertIn("privacy", result)

    def test_savings_report_human_text_no_db_unconfigured(self) -> None:
        stdout = io.StringIO()
        code = cli.agentflow_cli(
            ["savings", "report", "--config-dir", self.config_dir],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("AgentFlow savings report", output)
        self.assertIn("activation", output)

    def test_savings_report_json_with_configured_openai(self) -> None:
        # Activate OpenAI first
        cli.agentflow_cli(["activate", "openai", "--config-dir", self.config_dir], stdout=io.StringIO())
        stdout = io.StringIO()
        # Use a non-existent DB path so no store is opened
        code = cli.agentflow_cli(
            ["savings", "report", "--json", "--config-dir", self.config_dir,
             "--db", str(Path(self.config_dir) / "nonexistent.sqlite3")],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.savings_report.v1")
        # Claude is unconfigured so there should be an activation opportunity
        families = [o["opportunity_family"] for o in result["opportunities"]]
        self.assertIn(OPPORTUNITY_FAMILY_ACTIVATION, families)

    def test_savings_report_no_raw_ids_in_output(self) -> None:
        stdout = io.StringIO()
        cli.agentflow_cli(
            ["savings", "report", "--json", "--config-dir", self.config_dir],
            stdout=stdout,
        )
        output = stdout.getvalue()
        # The privacy summary keys are expected (session_ids_included, request_ids_included)
        # but actual raw session/request ID values must not appear.
        self.assertNotIn(_SECRET_SESSION, output)
        self.assertNotIn(_SECRET_REQUEST_ID, output)

    def test_savings_report_json_has_required_top_level_fields(self) -> None:
        stdout = io.StringIO()
        cli.agentflow_cli(
            ["savings", "report", "--json", "--config-dir", self.config_dir],
            stdout=stdout,
        )
        result = json.loads(stdout.getvalue())
        for field in ("schema", "ok", "generated_at", "privacy", "opportunities", "opportunity_count"):
            self.assertIn(field, result, f"Missing top-level field {field!r}")


if __name__ == "__main__":
    unittest.main()
