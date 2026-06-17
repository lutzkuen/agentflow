import asyncio
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import server, stats as stats_views
    from agentflow_proxy.dashboard_app import create_dashboard_app
    from agentflow_proxy import routing_experiments
    from agentflow_proxy.store import Store, stable_json, utc_now


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class StatsFullTest(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_tier_backoff_until = dict(server._tier_backoff_until)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server._tier_backoff_until.clear()
        server._tier_backoff_until.update(self.old_tier_backoff_until)

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

    def _assert_shadow_promotion_forbidden_absent(self, rendered: str) -> None:
        for forbidden in (
            "raw shadow readiness prompt",
            "raw shadow readiness response",
            "raw shadow failure prompt",
            "raw shadow failure response",
            "raw-primary-error-body",
            "raw-shadow-error-body",
            "req-shadow-secret",
            "session-shadow-secret",
            "tenant-shadow-secret",
            "account-shadow-secret",
            "/tmp/shadow-secret.py",
            "cache-shadow-secret",
            "tool-shadow-secret",
            "authorization-shadow-secret",
            "sk-shadow-secret",
            "shadow-call-secret",
            "routing reason raw secret",
            "workflow raw secret",
            '"raw_prompt"',
            '"raw_response"',
            '"provider_body"',
            '"request_id"',
            '"session_id"',
            '"tenant_id"',
            '"account_id"',
            '"file_path"',
            '"cache_key"',
            '"tool_payload"',
            '"authorization"',
            '"api_key"',
            '"secret"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_stats_full_exposes_active_crunch_rule_coverage_after_widening(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "crunch_rules.yaml")
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write(
                    """
enabled: true
request_shape_repeated_context_canaries:
  enabled: true
  schema: agentflow.request_shape_repeated_context_canaries.v1
  rules:
    - id: raw-policy-secret-should-not-leak
      enabled: true
      policy_source: local-manual
      cohort_id: raw-cohort-secret-should-not-leak
      rollout:
        canary_enabled: true
        canary_fraction: 0.2
        holdout_fraction: 0.1
      policy_decision:
        schema: agentflow.request_shape_crunch_policy_decision_rule_metadata.v1
        decision: widen
        graduation_decision: widen
        applied_count: 7
        holdout_count: 8
        skipped_count: 59
        blocked_count: 0
        observed_saved_chars: 42952
        observed_saved_tokens: 10738
        observed_saved_usd: 0.032214
        metadata_only: true
        aggregate_only: true
"""
                )
            with patch.dict(os.environ, {"AGENTFLOW_CRUNCH_RULES": rules_path}, clear=False):
                result = asyncio.run(stats_views.stats_full(server.store))

        coverage = result["active_crunch_rule_coverage"]
        self.assertEqual(coverage["schema"], "agentflow.active_crunch_rule_coverage.v1")
        self.assertEqual(coverage["status"], "observed")
        self.assertEqual(coverage["rule_file"], "crunch_rules.yaml")
        self.assertFalse(coverage["rules_path_included"])
        self.assertEqual(coverage["summary"]["active_rule_count"], 1)
        self.assertEqual(coverage["summary"]["widened_rule_count"], 1)
        self.assertEqual(coverage["summary"]["applied_count"], 7)
        self.assertEqual(coverage["summary"]["holdout_count"], 8)
        self.assertEqual(coverage["summary"]["skipped_count"], 59)
        self.assertEqual(coverage["summary"]["observed_saved_tokens"], 10738)
        self.assertAlmostEqual(coverage["summary"]["observed_saved_usd"], 0.032214)
        self.assertTrue(coverage["privacy"]["metadata_only"])
        self.assertTrue(coverage["privacy"]["aggregate_only"])
        self.assertFalse(coverage["privacy"]["raw_prompts_included"])
        rendered = json.dumps(coverage)
        self.assertNotIn(tmp, rendered)
        self.assertNotIn("raw-policy-secret-should-not-leak", rendered)
        self.assertNotIn("raw-cohort-secret-should-not-leak", rendered)

    def test_lightweight_stats_routing_rows_include_openai_canary_lifecycle_counts(self):
        observed_at = datetime.now(timezone.utc).isoformat()

        def log_openai_canary_call(call_id: str, *, cohort: str, routed_model: str, status_code: int = 200, retry_count: int = 0) -> None:
            canary = {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "status": "applied" if cohort == "canary_applied" else "holdout",
                "cohort": cohort,
                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": routed_model,
                "category": "chat",
            }
            routing = {
                "provider": "openai",
                "requested_model": "gpt-5.4",
                "routed_model": routed_model,
                "category": "chat",
                "openai_canary": canary,
            }
            if status_code >= 400:
                routing["fallback_reason"] = "rate_limited"
                canary["fallback_reason"] = "rate_limited"
            server.store.log_call(
                id=call_id,
                created_at=observed_at,
                path="/v1/responses",
                requested_model="gpt-5.4",
                routed_model=routed_model,
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=100,
                input_tokens_est=300,
                output_tokens_est=40,
                actual_input_tokens=300,
                actual_output_tokens=40,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="raw-openai-session-secret",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="openai",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
            )

        log_openai_canary_call("openai-canary-applied", cohort="canary_applied", routed_model="gpt-5.4-mini", status_code=500, retry_count=1)
        log_openai_canary_call("openai-canary-holdout", cohort="canary_holdout", routed_model="gpt-5.4")

        result = asyncio.run(stats_views.stats(server.store, self.tmp.name))
        rows = {
            (row["requested_model"], row["routed_model"]): row
            for row in result["routing"]
        }

        applied = rows[("gpt-5.4", "gpt-5.4-mini")]
        holdout = rows[("gpt-5.4", "gpt-5.4")]
        self.assertEqual(applied["source_surface"], "openai_responses")
        self.assertEqual(applied["endpoint"], "responses")
        self.assertEqual(applied["category"], "chat")
        self.assertEqual(applied["openai_canary_applied_count"], 1)
        self.assertEqual(applied["openai_canary_error_count"], 1)
        self.assertEqual(applied["openai_canary_retry_count"], 1)
        self.assertEqual(applied["openai_canary_fallback_count"], 1)
        self.assertEqual(applied["openai_canary_latest_observed_at"], observed_at)
        self.assertEqual(holdout["openai_canary_holdout_count"], 1)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-openai-session-secret", rendered)

    def test_lightweight_stats_routing_rows_include_anthropic_canary_lifecycle_counts(self):
        observed_at = datetime.now(timezone.utc).isoformat()

        def log_anthropic_canary_call(
            call_id: str,
            *,
            cohort: str,
            routed_model: str,
            status: str | None = None,
            reason: str | None = None,
            status_code: int = 200,
            retry_count: int = 0,
            fallback_reason: str | None = None,
        ) -> None:
            canary_status = status or {
                "canary_applied": "applied",
                "canary_holdout": "holdout",
                "safety_stopped": "safety_stopped",
                "skipped": "not_selected",
            }.get(cohort, "unknown")
            canary = {
                "enabled": True,
                "policy_id": "local-phase-sonnet-haiku-canary-v1",
                "status": canary_status,
                "cohort": cohort,
                "reason": reason or "selected-canary",
                "requested_model": "claude-sonnet-4-6",
                "target_model": "claude-haiku-4-5-20251001",
                "actual_forwarded_model": routed_model,
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "workflow_phase_confidence": "high",
                "source_surface": "anthropic_messages",
                "stream": True,
            }
            routing = {
                "provider": "anthropic",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": routed_model,
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "phase_canary": canary,
            }
            if fallback_reason:
                routing["fallback_reason"] = fallback_reason
                canary["fallback_reason"] = fallback_reason
            server.store.log_call(
                id=call_id,
                created_at=observed_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model=routed_model,
                stream=1,
                cache_hit=0,
                status_code=status_code,
                latency_ms=100,
                input_tokens_est=300,
                output_tokens_est=40,
                actual_input_tokens=300,
                actual_output_tokens=40,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="raw-anthropic-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
                requested_model_family="sonnet",
                routed_model_family="haiku" if "haiku" in routed_model else "sonnet",
            )

        log_anthropic_canary_call(
            "anthropic-canary-applied",
            cohort="canary_applied",
            routed_model="claude-haiku-4-5-20251001",
            status_code=500,
            retry_count=1,
            fallback_reason="rate_limited",
        )
        log_anthropic_canary_call(
            "anthropic-canary-holdout",
            cohort="canary_holdout",
            routed_model="claude-sonnet-4-6",
            reason="selected-holdout",
        )
        log_anthropic_canary_call(
            "anthropic-canary-safety",
            cohort="safety_stopped",
            routed_model="claude-sonnet-4-6",
            reason="thinking-safety-gate",
        )

        result = asyncio.run(stats_views.stats(server.store, self.tmp.name))
        rows = {
            (row["requested_model"], row["routed_model"]): row
            for row in result["routing"]
        }

        applied = rows[("claude-sonnet-4-6", "claude-haiku-4-5-20251001")]
        requested = rows[("claude-sonnet-4-6", "claude-sonnet-4-6")]
        self.assertEqual(applied["source_surface"], "anthropic_messages")
        self.assertEqual(applied["endpoint"], "messages")
        self.assertEqual(applied["category"], "tool-result")
        self.assertEqual(applied["anthropic_canary_applied_count"], 1)
        self.assertEqual(applied["anthropic_canary_error_count"], 1)
        self.assertEqual(applied["anthropic_canary_retry_count"], 1)
        self.assertEqual(applied["anthropic_canary_fallback_count"], 1)
        self.assertEqual(applied["anthropic_canary_latest_observed_at"], observed_at)
        self.assertEqual(requested["anthropic_canary_holdout_count"], 1)
        self.assertEqual(requested["anthropic_canary_safety_stopped_count"], 1)
        self.assertEqual(requested["anthropic_canary_latest_observed_at"], observed_at)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-anthropic-session-secret", rendered)

    def test_crunch_savings_uses_cache_blended_input_rate(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=1_000,
            output_tokens_est=0,
            actual_input_tokens=1_000,
            actual_output_tokens=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 1_000}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-a",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=9_000,
            retry_count=0,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]

        self.assertAlmostEqual(summary["crunch_savings_usd"], 0.00057, places=6)
        self.assertAlmostEqual(summary["today_crunch_savings_usd"], 0.00057, places=6)

    def test_executive_health_errors_use_today_boundary(self):
        today = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        def log_call(suffix: str, created_at: datetime, status_code: int) -> None:
            server.store.log_call(
                id=f"health-errors-{suffix}",
                created_at=created_at.isoformat(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=10,
                input_tokens_est=10,
                output_tokens_est=10,
                actual_input_tokens=10,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.001,
                crunch_json=stable_json({"changed": False}),
                routing_json=None,
                cache_json=None,
                error="HTTP error" if status_code >= 400 else None,
                request_json=None,
                response_json=None,
                session_id=f"session-{suffix}",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        log_call("today-error", today, 500)
        log_call("yesterday-error", yesterday, 500)
        log_call("today-success", today, 200)

        result = asyncio.run(stats_views.stats_full(server.store))
        health = result["executive_summary"]["health"]

        self.assertEqual(health["errors"], 1)
        self.assertEqual(health["errors_today"], 1)
        self.assertEqual(health["total_errors"], 2)
        self.assertEqual(sum(row["count"] for row in result["today_error_breakdown"]), 1)
        self.assertEqual(sum(row["count"] for row in result["error_breakdown"]), 2)

    def test_policy_state_exposes_codex_app_surface_cache_disabled(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "1",
                "AGENTFLOW_CODEX_APP_CACHE": "0",
                "AGENTFLOW_CODEX_APP_UPSTREAM": "ws://127.0.0.1:4999",
            },
            clear=False,
        ):
            result = asyncio.run(stats_views.stats_policies())

        self.assertIn("routing", result)
        self.assertIn("crunch", result)
        self.assertIn("cache", result)
        surface = result["source_surfaces"]["codex_turn"]
        self.assertTrue(surface["optimization"]["enabled"])
        self.assertFalse(surface["cache"]["enabled"])
        self.assertFalse(surface["cache"]["exact_cache"]["enabled"])
        self.assertEqual(surface["cache"]["disabled_reason"], "AGENTFLOW_CODEX_APP_CACHE is not 1")
        self.assertEqual(surface["cache"]["exact_cache"]["upstream"], "ws://127.0.0.1:4999")
        self.assertIn("input", surface["safe_turn_params"]["allowed_keys"])
        self.assertEqual(surface["action_like_skip_behavior"]["reason"], "action-like-params")
        self.assertEqual(surface["routing"]["policy_source"], result["routing"]["policy_source"])
        self.assertEqual(surface["crunch"]["rule_path"], result["crunch"]["rule_path"])
        self.assertEqual(surface["cache"]["rule_path"], result["cache"]["rule_path"])
        self.assertEqual(surface["managed_optimizer_required"], False)
        self.assertIn("codex_app", result)
        self.assertFalse(result["codex_app"]["review_only"])
        self.assertIn("file", result["codex_app"])
        self.assertEqual(result["summary"]["policy_count"], 5)
        self.assertEqual(result["summary"]["source_surface_policy_count"], 1)

    def test_policy_state_exposes_codex_app_surface_cache_enabled(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "0",
                "AGENTFLOW_CODEX_APP_CACHE": "1",
                "AGENTFLOW_CODEX_APP_CACHE_NAMESPACE": "codex-test",
            },
            clear=False,
        ):
            result = asyncio.run(stats_views.stats_policies())

        surface = result["source_surfaces"]["codex_turn"]
        self.assertFalse(surface["optimization"]["enabled"])
        self.assertTrue(surface["cache"]["enabled"])
        self.assertTrue(surface["cache"]["exact_cache"]["enabled"])
        self.assertEqual(surface["cache"]["exact_cache"]["namespace"], "codex-test")
        self.assertEqual(surface["cache"]["exact_cache"]["provider"], "codex-app")
        self.assertEqual(surface["cache"]["exact_cache"]["cache_url"], "codex-app://turn/start")
        self.assertEqual(surface["cache"]["exact_cache"]["replayability_level"], "local-exact-response")

    def test_codex_canary_impact_reports_rule_candidate_counts_and_privacy(self):
        rule_meta = {
            "schema": "agentflow.codex_app_rule_execution.v1",
            "rule_id": "codex-rule-a",
            "candidate_id": "candidate-a",
            "policy_id": "policy-a",
            "policy_source": "managed-recommended",
            "condition_keys": ["workflow_phase"],
            "action_keys": ["model_hint", "cache_eligible"],
            "raw_conditions_included": False,
            "raw_actions_included": False,
            "raw_params_included": False,
        }

        def log_turn(suffix, routing, cache, *, error_code=None, latency=100, result_chars=80):
            server.store.log_codex_app_event(
                id=f"start-{suffix}",
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=f"raw-request-{suffix}",
                thread_id=f"raw-thread-{suffix}",
                session_id=f"raw-session-{suffix}",
                message_chars=100,
                params_chars=80,
                input_items=1,
                input_text_chars=400,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                routing_json=stable_json(routing),
                crunch_json=stable_json({"status": "skipped", "policy_source": "managed-recommended"}),
                cache_json=stable_json(cache),
            )
            server.store.log_codex_app_event(
                id=f"end-{suffix}",
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=f"raw-request-{suffix}",
                thread_id=f"raw-thread-{suffix}",
                session_id=f"raw-session-{suffix}",
                message_chars=120,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=error_code,
                error_message="raw error body must not leak" if error_code else None,
                latency_ms=latency,
            )

        applied_routing = {
            "status": "applied",
            "reason": "codex-app-rule-canary-applied",
            "applied": True,
            "canary": "codex-app-rule",
            "canary_cohort": "canary_applied",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "target_model": "gpt-5.4-mini",
            "policy_source": "managed-recommended",
            "codex_app_rule": rule_meta,
        }
        holdout_routing = {
            **applied_routing,
            "status": "skipped",
            "reason": "codex-app-rule-canary-holdout",
            "applied": False,
            "canary_cohort": "canary_holdout",
            "routed_model": "gpt-5.4",
        }
        stopped_routing = {
            **applied_routing,
            "status": "safety_stopped",
            "reason": "local-canary-safety-stop",
            "applied": False,
            "canary_cohort": "safety_stopped",
            "safety_stop": {
                "tripped": True,
                "reason_codes": ["applied-error-rate-above-threshold"],
                "rule_id": "codex-rule-a",
                "candidate_id": "candidate-a",
            },
        }
        applied_cache = {
            "status": "miss",
            "reason": "exact-miss",
            "eligible": True,
            "canary": "codex-app-rule",
            "canary_cohort": "canary_applied",
            "outcome_bucket": "miss",
            "policy_source": "managed-recommended",
            "codex_app_rule": rule_meta,
        }
        holdout_cache = {
            **applied_cache,
            "status": "holdout",
            "reason": "codex-app-rule-canary-holdout",
            "canary_cohort": "canary_holdout",
            "outcome_bucket": "holdout",
        }
        invalidated_cache = {
            **applied_cache,
            "status": "miss",
            "reason": "dependency-changed",
            "outcome_bucket": "invalidated",
        }
        log_turn("applied", applied_routing, applied_cache, latency=100)
        log_turn("holdout", holdout_routing, holdout_cache, latency=60)
        log_turn("stopped", stopped_routing, invalidated_cache, error_code=500, latency=130)

        result = asyncio.run(stats_views.stats_codex_canary_impact(server.store, limit=10))

        self.assertEqual(result["schema"], "agentflow.codex_app_canary_impact_by_rule.v1")
        self.assertEqual(result["summary"]["rule_candidate_count"], 2)
        self.assertEqual(result["summary"]["applied_count"], 3)
        self.assertEqual(result["summary"]["holdout_count"], 2)
        self.assertEqual(result["summary"]["safety_stopped_count"], 1)
        routing = next(row for row in result["rules"] if row["action_family"] == "routing")
        cache = next(row for row in result["rules"] if row["action_family"] == "cache")
        self.assertEqual(routing["rule_id"], "codex-rule-a")
        self.assertEqual(routing["candidate_id"], "candidate-a")
        self.assertEqual(routing["policy_source"], "managed-recommended")
        self.assertEqual(routing["applied_count"], 1)
        self.assertEqual(routing["holdout_count"], 1)
        self.assertEqual(routing["error_count"], 1)
        self.assertEqual(routing["safety_stopped_count"], 1)
        self.assertEqual(routing["applied_minus_holdout_latency_avg_ms"], 40)
        self.assertEqual(cache["cache"]["miss_count"], 2)
        self.assertEqual(cache["cache"]["holdout_count"], 1)
        self.assertEqual(cache["cache"]["invalidation_count"], 1)
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-request", encoded)
        self.assertNotIn("raw-session", encoded)
        self.assertNotIn("raw error body", encoded)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])

    def test_full_stats_include_executive_summary_for_top_dashboard_tiles(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=1,
            status_code=200,
            latency_ms=10,
            input_tokens_est=1_200,
            output_tokens_est=120,
            actual_input_tokens=1_000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 500}),
            routing_json=stable_json({"reason": "test route"}),
            cache_json=stable_json({"status": "hit", "reason": "exact-match", "hit_type": "exact"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-exec",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=25,
            provider="anthropic",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-exec",
            thread_id="thread-exec",
            message_chars=200,
            params_chars=50,
            input_items=1,
            input_text_chars=123,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-exec",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-exec",
            thread_id="thread-exec",
            message_chars=120,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=2000,
            session_id="codex-exec",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        executive = result["executive_summary"]

        self.assertEqual(executive["schema"], "agentflow.executive_summary.v1")
        self.assertEqual(executive["tokens_today"]["provider_input_tokens"], 3_000)
        self.assertEqual(executive["tokens_today"]["provider_output_tokens"], 100)
        self.assertEqual(executive["tokens_today"]["provider_total_tokens"], 3_100)
        self.assertEqual(executive["tokens_today"]["codex_app_turns"], 1)
        self.assertEqual(executive["tokens_today"]["codex_app_input_text_chars"], 123)
        self.assertEqual(executive["tokens_today"]["codex_app_input_tokens_est"], 30)
        self.assertEqual(executive["tokens_today"]["codex_app_output_tokens_est"], 20)
        self.assertEqual(executive["tokens_today"]["codex_app_total_tokens_est"], 50)
        self.assertEqual(executive["tokens_today"]["total_tokens"], 3_150)
        self.assertTrue(executive["tokens_today"]["codex_app_cost_known"])
        self.assertTrue(executive["tokens_today"]["codex_app_cost_estimated"])
        self.assertEqual(executive["tokens_today"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(executive["spend"]["today_provider_spend_usd"], 0.001, places=6)
        self.assertIn(
            executive["tokens_today"]["codex_app_pricing_basis"]["model"],
            {"gpt-5.5", "gpt-5.3-codex"},
        )
        self.assertGreater(executive["spend"]["today_codex_app_estimated_spend_usd"], 0.0)
        self.assertAlmostEqual(
            executive["spend"]["today_calculated_spend_usd"],
            executive["spend"]["today_provider_spend_usd"] + executive["spend"]["today_codex_app_estimated_spend_usd"],
            places=6,
        )
        self.assertEqual(executive["spend"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertGreater(executive["spend"]["thinking_cost_today_usd"], 0)
        savings = executive["savings"]
        buckets = savings["today_buckets"]
        agentflow_buckets = savings["today_agentflow_generated_buckets"]
        self.assertIn("routing_usd", buckets)
        self.assertIn("crunching_usd", buckets)
        self.assertAlmostEqual(buckets["exact_local_cache_usd"], 0.003, places=6)
        self.assertIn("provider_prompt_cache_discount_usd", buckets)
        # New split fields: provider prompt-cache is not in agentflow_generated_buckets
        self.assertNotIn("provider_prompt_cache_discount_usd", agentflow_buckets)
        self.assertIn("routing_usd", agentflow_buckets)
        self.assertIn("crunching_usd", agentflow_buckets)
        self.assertAlmostEqual(agentflow_buckets["exact_local_cache_usd"], 0.003, places=6)
        self.assertIn("today_agentflow_generated_savings_usd", savings)
        self.assertIn("provider_prompt_cache_discount_usd", savings)
        self.assertIn("provider_prompt_cache_economics", savings)
        self.assertFalse(executive["hard_floor"]["excludes_unknown_codex_app_cost"])
        self.assertTrue(executive["hard_floor"]["codex_app_cost_estimated"])
        self.assertLessEqual(
            executive["hard_floor"]["today_unavoidable_provider_spend_usd"],
            executive["spend"]["today_baseline_calculated_cost_usd"],
        )
        self.assertIn("accounting_today", executive)
        self.assertIn("source_surfaces", executive["accounting_today"])
        json.dumps(executive)

    def test_weekly_stats_include_codex_turns_and_zero_daily_rows(self):
        days = stats_views._utc_day_window(7)
        provider_day = days[-3]
        empty_day = days[-2]
        today = days[-1]

        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=f"{provider_day}T12:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="provider-session",
            category="chat",
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=f"{today}T09:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="weekly-codex",
            thread_id="thread-weekly",
            message_chars=500,
            params_chars=20,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-weekly",
            cache_json=stable_json({"status": "miss", "reason": "codex-app-cache-disabled"}),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=f"{today}T09:00:02+00:00",
            direction="server_to_client",
            method="turn/completed",
            request_id="weekly-codex",
            thread_id="thread-weekly",
            message_chars=80,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=2000,
            session_id="codex-weekly",
        )

        result = asyncio.run(stats_views.stats_weekly(server.store))

        self.assertEqual(result["schema"], "agentflow.weekly_activity.v1")
        self.assertIn("generated_at", result)
        self.assertEqual([row["day"] for row in result["days"]], days)
        self.assertEqual(len(result["days"]), 7)

        by_day = {row["day"]: row for row in result["days"]}
        provider_row = by_day[provider_day]
        self.assertEqual(provider_row["provider_calls"], 1)
        self.assertEqual(provider_row["codex_turns"], 0)
        self.assertEqual(provider_row["total_calls"], 1)
        self.assertEqual(provider_row["provider_tokens"], 160)
        self.assertAlmostEqual(provider_row["cost_est_usd"], 0.001, places=6)
        self.assertAlmostEqual(provider_row["savings_usd"], 0.002, places=6)

        zero_row = by_day[empty_day]
        self.assertEqual(zero_row["total_units"], 0)
        self.assertEqual(zero_row["provider_calls"], 0)
        self.assertEqual(zero_row["codex_turns"], 0)
        self.assertEqual(zero_row["total_tokens"], 0)
        self.assertEqual(zero_row["cost_est_usd"], 0.0)

        today_row = by_day[today]
        expected_codex = stats_views._codex_estimates_with_cache(400, 80, {"status": "miss"})
        self.assertEqual(today_row["provider_calls"], 0)
        self.assertEqual(today_row["codex_turns"], 1)
        self.assertEqual(today_row["total_calls"], 1)
        self.assertEqual(today_row["successful_calls"], 1)
        self.assertEqual(today_row["codex_tokens_est"], expected_codex["total_tokens_est"])
        self.assertEqual(today_row["total_tokens"], expected_codex["total_tokens_est"])
        self.assertAlmostEqual(today_row["codex_cost_est_usd"], expected_codex["cost_est_usd"], places=6)
        self.assertAlmostEqual(today_row["cost_est_usd"], expected_codex["cost_est_usd"], places=6)
        self.assertEqual(today_row["avg_latency_ms"], 2000)
        self.assertEqual(today_row["cost_basis"], "provider-reported + codex-estimated-from-chars")

        totals = result["totals"]
        self.assertEqual(totals["provider_calls"], 1)
        self.assertEqual(totals["codex_turns"], 1)
        self.assertEqual(totals["total_units"], 2)
        self.assertEqual(totals["total_calls"], 2)
        self.assertEqual(totals["provider_tokens"], 160)
        self.assertEqual(totals["codex_tokens_est"], expected_codex["total_tokens_est"])
        self.assertEqual(totals["total_tokens"], 160 + expected_codex["total_tokens_est"])

    def test_dashboard_weekly_table_exposes_provider_and_codex_columns(self):
        html = stats_views.dashboard_html()

        self.assertIn("<h2>7-day activity statistics</h2>", html)
        self.assertIn('<th data-sort-type="number">Provider calls</th>', html)
        self.assertIn('<th data-sort-type="number">Codex turns</th>', html)
        self.assertIn('<th data-sort-type="number">Tokens</th>', html)
        self.assertIn('<th data-sort-type="text">Cost basis</th>', html)
        self.assertIn("row.codex_turns", html)
        self.assertIn("row.codex_tokens_est", html)

    def test_managed_recommendation_stats_cover_recent_statuses_and_feedback(self):
        saved_enabled = os.environ.get("AGENTFLOW_RECOMMENDATION_ENABLED")
        saved_url = os.environ.get("AGENTFLOW_RECOMMENDATION_SERVER_URL")
        os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.local"

        def log_call(created_at, routing_json):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=10,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing_json) if routing_json is not None else None,
                cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="managed-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        try:
            log_call(
                "2026-06-08T10:00:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": False,
                        "server_url": "http://managed.local",
                        "status": "skipped",
                        "reason": "disabled",
                        "fallback": "local-policy",
                        "applied": False,
                        "outcome_feedback": {
                            "enabled": False,
                            "status": "skipped",
                            "reason": "disabled",
                        },
                    }
                },
            )
            log_call(
                "2026-06-08T10:01:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": True,
                        "server_url": "http://managed.local",
                        "status": "received",
                        "reason": "candidate matched",
                        "policy_source": "managed-recommended",
                        "policy_id": "candidate-route-chat",
                        "target_model": "claude-haiku-4-5-20251001",
                        "applied": True,
                        "changed_model": True,
                        "latency_ms": 120,
                        "outcome_feedback": {
                            "enabled": True,
                            "status": "sent",
                            "reason": "accepted",
                            "latency_ms": 30,
                            "optimization_unit_id": 42,
                        },
                    }
                },
            )
            log_call(
                "2026-06-08T10:02:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": True,
                        "server_url": "http://managed.local",
                        "status": "error",
                        "reason": "server-error",
                        "fallback": "local-policy",
                        "applied": False,
                        "latency_ms": 50,
                        "error": '{"error":{"type":"managed_server_error","message":"down"}}',
                        "outcome_feedback": {
                            "enabled": True,
                            "status": "error",
                            "reason": "request-failed",
                            "latency_ms": 20,
                            "error": "RuntimeError('feedback unavailable')",
                        },
                    }
                },
            )
            log_call("2026-06-08T10:03:00+00:00", None)

            result = asyncio.run(stats_views.stats_managed_recommendations(server.store, limit=20))
        finally:
            if saved_enabled is None:
                os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
            else:
                os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = saved_enabled
            if saved_url is None:
                os.environ.pop("AGENTFLOW_RECOMMENDATION_SERVER_URL", None)
            else:
                os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = saved_url

        summary = result["summary"]
        self.assertEqual(result["schema"], "agentflow.managed_recommendations.v1")
        self.assertFalse(result["current_config"]["enabled"])
        self.assertIn("local policy remains authoritative", result["current_config"]["offline_state"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertEqual(summary["window_calls"], 4)
        self.assertEqual(summary["metadata_rows"], 4)
        self.assertEqual(summary["historical_null_rows"], 0)
        self.assertEqual(summary["disabled_count"], 2)
        self.assertEqual(summary["enabled_count"], 2)
        self.assertEqual(summary["received_count"], 1)
        self.assertEqual(summary["applied_count"], 1)
        self.assertEqual(summary["changed_model_count"], 1)
        self.assertAlmostEqual(summary["observed_savings_usd"], 0.002, places=6)
        self.assertAlmostEqual(summary["applied_observed_savings_usd"], 0.002, places=6)
        self.assertAlmostEqual(summary["changed_model_observed_savings_usd"], 0.002, places=6)
        self.assertEqual(summary["positive_savings_count"], 1)
        self.assertEqual(summary["observed_savings_basis"], "calls.cost_baseline_usd-minus-cost_est_usd")
        self.assertEqual(summary["observed_savings_attribution"], "managed-recommendation-model-change")
        self.assertEqual(summary["server_error_count"], 1)
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(summary["feedback_sent_count"], 1)
        self.assertEqual(summary["feedback_skipped_count"], 1)
        self.assertEqual(summary["feedback_failed_count"], 1)
        self.assertEqual(summary["feedback_sanitized_count"], 3)
        self.assertEqual(summary["avg_recommendation_latency_ms"], 85)
        self.assertEqual(summary["avg_feedback_latency_ms"], 25)
        self.assertEqual(summary["last_recommendation_error_class"], "server-error")
        self.assertEqual(summary["last_feedback_error_class"], "request-failed")
        self.assertEqual({row["value"]: row["count"] for row in result["policy_ids"]}, {"candidate-route-chat": 1})
        self.assertEqual(result["recent"][0]["recommendation_status"], "skipped-local-blocker")
        self.assertEqual(result["recent"][0]["recommendation_reason"], "policy-decision-metadata-missing")
        applied_recent = next(row for row in result["recent"] if row["applied"])
        self.assertAlmostEqual(applied_recent["observed_savings_usd"], 0.002, places=6)
        self.assertTrue(applied_recent["observed_savings_attributed_to_managed"])
        json.dumps(result)

    def test_managed_recommendation_dashboard_endpoint_and_panel_render_without_server(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AGENTFLOW_POLICY_EVENTS_LOG": os.path.join(tmp, "policy_events.jsonl")},
            clear=False,
        ):
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "fetch-review",
                ok=True,
                details={
                    "source": "cli",
                    "recommendation_health": {
                        "schema": "agentflow.recommendation_health.v1",
                        "status": "warning",
                        "warning_count": 1,
                        "rows": [
                            {
                                "kind": "stale_evidence",
                                "code": "stale-evidence",
                                "candidate_id": "candidate-route-chat",
                                "details": {"sample_count": 24, "last_seen_at": "2026-06-01T12:00:00+00:00"},
                            }
                        ],
                        "privacy": {"metadata_only": True, "raw_prompts_included": False},
                    },
                },
            )

            app = create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.anthropic.com",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)

            stats_response = client.get("/agentflow/stats/managed-recommendations")
            self.assertEqual(stats_response.status_code, 200)
            payload = stats_response.json()
            self.assertEqual(payload["schema"], "agentflow.managed_recommendations.v1")
            self.assertEqual(payload["current_config"]["mode"], "local-only")
            self.assertFalse(payload["current_config"]["enabled"])
            self.assertEqual(
                payload["current_config"]["offline_state"],
                "managed recommendations disabled; local policy remains authoritative",
            )
            self.assertEqual(
                payload["recommendation_health"]["latest_fetch_review"]["rows"][0]["candidate_id"],
                "candidate-route-chat",
            )

            html = client.get("/agentflow/dashboard")
            self.assertEqual(html.status_code, 200)
            self.assertIn("Managed recommendation status", html.text)
            self.assertIn("Managed recommendation health", html.text)
            self.assertIn("/agentflow/stats/managed-recommendations", html.text)
            self.assertIn("managed-summary-tbody", html.text)
            self.assertIn("managed-health-tbody", html.text)

    def test_openai_scoreboard_answers_go_no_go_from_metadata_and_suppresses_claude_no_traffic(self):
        def log_openai_call(created_at, *, routing, status_code=200, latency_ms=100, retry_count=0, cache=None, crunch=None, actual=0.02, baseline=0.02):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/responses",
                requested_model="gpt-5-codex",
                routed_model=routing.get("routed_model", "gpt-5-codex"),
                stream=0,
                cache_hit=1 if (cache or {}).get("status") == "hit" else 0,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=900,
                actual_output_tokens=90,
                cost_est_usd=actual,
                cost_baseline_usd=baseline,
                crunch_json=stable_json(crunch or {"changed": False, "tokens_saved_est": 0}),
                routing_json=stable_json(routing),
                cache_json=stable_json(cache or {"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="openai-secret-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                thinking_output_tokens=0,
                provider="openai",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model_family="gpt-5-codex",
                routed_model_family="gpt-5-codex",
            )

        risk = {
            "error_rate": 0.01,
            "retry_rate": 0.02,
            "fallback_rate": 0.01,
            "latency_regression_ratio": 0.9,
            "missing_fields": [],
        }
        log_openai_call(
            "2026-06-09T10:00:00+00:00",
            routing={
                "routed_model": "gpt-5-codex",
                "managed_recommendation": {
                    "mode": "observe-only",
                    "status": "skipped",
                    "enabled": False,
                    "applied": False,
                    "raw_payload_included": False,
                },
            },
        )
        log_openai_call(
            "2026-06-09T10:01:00+00:00",
            routing={
                "routed_model": "gpt-5-codex",
                "managed_recommendation": {
                    "mode": "dry-run",
                    "status": "dry-run",
                    "lifecycle_event": "dry_run",
                    "policy_id": "openai-mini-candidate",
                    "target_model": "gpt-5-mini",
                    "would_route_model": "gpt-5-mini",
                    "projection": {"projected_input_savings_usd": 0.01, "risk": risk},
                },
            },
            retry_count=1,
            crunch={"changed": True, "tokens_saved_est": 25},
        )
        log_openai_call(
            "2026-06-09T10:02:00+00:00",
            routing={
                "routed_model": "gpt-5-mini",
                "managed_recommendation": {
                    "mode": "canary",
                    "status": "applied",
                    "lifecycle_event": "canary_applied",
                    "policy_id": "openai-mini-candidate",
                    "target_model": "gpt-5-mini",
                    "applied": True,
                    "changed_model": True,
                    "projection": {"projected_input_savings_usd": 0.01, "risk": risk},
                },
            },
            latency_ms=90,
            cache={"status": "hit", "reason": "exact-hit", "policy_source": "local-default"},
            actual=0.01,
            baseline=0.03,
        )
        log_openai_call(
            "2026-06-09T10:03:00+00:00",
            routing={
                "routed_model": "gpt-5-codex",
                "managed_recommendation": {
                    "mode": "canary",
                    "status": "holdout",
                    "lifecycle_event": "holdout",
                    "policy_id": "openai-mini-candidate",
                    "target_model": "gpt-5-mini",
                    "would_route_model": "gpt-5-mini",
                    "projection": {"projected_input_savings_usd": 0.01, "risk": risk},
                },
            },
            latency_ms=110,
        )
        log_openai_call(
            "2026-06-09T10:04:00+00:00",
            routing={
                "routed_model": "gpt-5-codex",
                "managed_recommendation": {
                    "mode": "canary",
                    "status": "skipped",
                    "lifecycle_event": "fallback",
                    "fallback": "local-policy",
                    "apply_reason": "provider-mismatch",
                    "policy_id": "wrong-provider-candidate",
                    "target_model": "claude-haiku-4-5-20251001",
                    "projection": {"projected_input_savings_usd": 0.0, "risk": risk},
                },
            },
        )

        result = asyncio.run(stats_views.stats_openai_scoreboard(server.store, limit=20))

        self.assertEqual(result["schema"], "agentflow.openai_optimization_scoreboard.v1")
        self.assertEqual(result["answer"], "helping")
        self.assertEqual(result["summary"]["openai_call_count"], 5)
        self.assertEqual(result["summary"]["retry_count"], 1)
        self.assertEqual(result["summary"]["fallback_count"], 1)
        self.assertEqual(result["summary"]["cache_hit_count"], 1)
        self.assertEqual(result["summary"]["tokens_saved_est"], 25)
        self.assertAlmostEqual(result["summary"]["observed_cost_savings_usd"], 0.02, places=6)
        self.assertAlmostEqual(result["summary"]["projected_cost_savings_usd"], 0.03, places=6)
        self.assertEqual(result["summary"]["observed_latency_delta_ms"], -20)
        self.assertEqual(
            result["companion_sections"]["anthropic_recommendations"]["status"],
            "no-traffic",
        )
        self.assertEqual(
            result["companion_sections"]["anthropic_recommendations"]["display"],
            "suppressed",
        )
        states = {row["value"]: row["count"] for row in result["state_breakdown"]}
        self.assertEqual(states["observed-only"], 1)
        self.assertEqual(states["dry-run"], 1)
        self.assertEqual(states["canary-applied"], 1)
        self.assertEqual(states["holdout"], 1)
        self.assertEqual(states["fallback"], 1)
        gates = {row["value"]: row["count"] for row in result["quality_gate_breakdown"]}
        self.assertEqual(gates["passed-local-gates"], 3)
        self.assertEqual(gates["failed-provider-target"], 1)
        candidate = next(row for row in result["candidates"] if row["candidate_id"] == "openai-mini-candidate")
        self.assertEqual(candidate["calls"], 3)
        self.assertEqual(candidate["observed_latency_delta_ms"], -20)
        rendered = json.dumps(result)
        self.assertNotIn("openai-secret-session", rendered)
        for key in ("raw_prompts_included", "raw_responses_included", "tool_bodies_included", "request_ids_included", "tenant_ids_included", "secrets_included"):
            self.assertFalse(result["privacy"][key])
        self.assertFalse(result["privacy"]["provider_calls_made"])
        json.dumps(result)

    def test_openai_scoreboard_dashboard_endpoint_panel_and_cli_are_metadata_only(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=50,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json({
                "managed_recommendation": {
                    "mode": "canary",
                    "status": "applied",
                    "lifecycle_event": "canary_applied",
                    "policy_id": "openai-mini-candidate",
                    "target_model": "gpt-5-mini",
                    "applied": True,
                    "projection": {
                        "projected_input_savings_usd": 0.002,
                        "risk": {
                            "error_rate": 0.0,
                            "retry_rate": 0.0,
                            "fallback_rate": 0.0,
                            "latency_regression_ratio": 0.8,
                            "missing_fields": [],
                        },
                    },
                }
            }),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="hidden-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5-codex",
            routed_model_family="gpt-5-mini",
        )
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.openai.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        client = TestClient(app)

        stats_response = client.get("/agentflow/stats/openai-scoreboard")
        self.assertEqual(stats_response.status_code, 200)
        payload = stats_response.json()
        self.assertEqual(payload["schema"], "agentflow.openai_optimization_scoreboard.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 1)
        self.assertFalse(payload["privacy"]["provider_calls_made"])

        html = client.get("/agentflow/dashboard")
        self.assertEqual(html.status_code, 200)
        self.assertIn("OpenAI optimization scoreboard", html.text)
        self.assertIn("/agentflow/stats/openai-scoreboard", html.text)
        self.assertIn("openai-scoreboard-summary-tbody", html.text)
        self.assertIn("openai-scoreboard-candidates-tbody", html.text)
        self.assertIn("Claude recommendation traffic state", html.text)

        from agentflow_proxy import cli

        output = io.StringIO()
        exit_code = cli.openai_scoreboard_cli(["--db", self.tmp.name, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        cli_payload = json.loads(output.getvalue())
        self.assertEqual(cli_payload["schema"], "agentflow.openai_optimization_scoreboard.v1")
        self.assertEqual(cli_payload["summary"]["openai_call_count"], 1)
        self.assertNotIn("hidden-session", output.getvalue())

    def test_openai_optimization_readiness_reports_conflicts_without_raw_content(self):
        from agentflow_proxy.openai_optimization_governor import attach_openai_optimization_governor

        routing = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "reason": "selected-canary",
            "text_chars": 42000,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "policy_source": "local-manual",
            "request_id": "req-unified-secret",
            "prompt": "raw unified readiness prompt",
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "reason": "selected-canary",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
                "policy_source": "local-manual",
            },
        }
        summary = {
            "schema": "agentflow.openai_old_context_summary.v1",
            "enabled": True,
            "status": "applied",
            "applied": True,
            "reason_codes": ["applied"],
            "policy_source": "local-manual",
            "summary": "raw unified summary text",
        }
        crunch = {
            "changed": True,
            "old_context_summarization": summary,
            "messages": [{"content": "raw unified readiness prompt"}],
        }
        cache = {
            "status": "hit",
            "reason": "exact-hit",
            "policy_source": "local-default",
            "cache_key": "cache-unified-secret",
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "canary_cohort": "canary_applied",
                "policy_source": "local-manual",
            },
        }
        attach_openai_optimization_governor(
            routing_meta=routing,
            crunch_meta=crunch,
            cache_meta=cache,
            summary_meta=summary,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            session_id="raw-unified-session",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            stream=0,
            cache_hit=1,
            status_code=200,
            latency_ms=80,
            input_tokens_est=1000,
            output_tokens_est=100,
            actual_input_tokens=900,
            actual_output_tokens=90,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json(crunch),
            routing_json=stable_json(routing),
            cache_json=stable_json(cache),
            error=None,
            request_json='{"input":"raw unified readiness prompt","request_id":"req-unified-secret","file_path":"/tmp/unified-secret.py"}',
            response_json='{"output_text":"raw unified response"}',
            session_id="raw-unified-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5-mini",
        )

        result = asyncio.run(stats_views.stats_openai_optimization_readiness(server.store, limit=20))

        self.assertEqual(result["schema"], "agentflow.openai_optimization_readiness.v1")
        self.assertTrue(result["read_only"])
        self.assertEqual(result["state"], "conflicts-observed")
        self.assertEqual(result["summary"]["openai_call_count"], 1)
        self.assertEqual(result["summary"]["governor_metadata_row_count"], 1)
        self.assertEqual(result["summary"]["selected_call_count"], 1)
        self.assertEqual(result["summary"]["conflicting_call_count"], 1)
        self.assertEqual(result["summary"]["suppressed_family_count"], 2)
        selected = {row["value"]: row["count"] for row in result["selected_family_breakdown"]}
        self.assertEqual(selected["routing"], 1)
        reasons = {row["value"]: row["count"] for row in result["suppression_reason_breakdown"]}
        self.assertEqual(reasons["conflicts-with-selected-family"], 2)
        family_rows = {row["family"]: row for row in result["families"]}
        self.assertEqual(family_rows["routing"]["selected_count"], 1)
        self.assertEqual(family_rows["old_context_summary"]["suppressed_count"], 1)
        self.assertEqual(family_rows["cache_replay"]["suppressed_count"], 1)
        self.assertEqual(result["recent_conflicts"][0]["selected_action_family"], "routing")
        self.assertEqual(
            {item["family"] for item in result["recent_conflicts"][0]["suppressed_families"]},
            {"old_context_summary", "cache_replay"},
        )
        self.assertFalse(result["privacy"]["provider_calls_made"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["raw_session_ids_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "raw unified readiness prompt",
            "raw unified summary text",
            "raw unified response",
            "req-unified-secret",
            "cache-unified-secret",
            "/tmp/unified-secret.py",
            "raw-unified-session",
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
            '"messages"',
            '"prompt"',
        ):
            self.assertNotIn(forbidden, rendered)

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.openai.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        client = TestClient(app)
        stats_response = client.get("/agentflow/stats/openai-optimization-readiness?limit=20")
        self.assertEqual(stats_response.status_code, 200)
        self.assertEqual(stats_response.json()["summary"]["conflicting_call_count"], 1)
        html = client.get("/agentflow/dashboard")
        self.assertEqual(html.status_code, 200)
        self.assertIn("OpenAI optimization readiness", html.text)
        self.assertIn("/agentflow/stats/openai-optimization-readiness", html.text)
        self.assertIn("openai-optimization-readiness-summary-tbody", html.text)
        self.assertIn("openai-optimization-readiness-families-tbody", html.text)
        self.assertIn("openai-optimization-readiness-conflicts-tbody", html.text)
        self.assertNotIn("raw-unified-session", html.text)

    def test_openai_canary_readiness_endpoint_uses_impact_fixture_metadata(self):
        def log_canary_call(call_id, *, cohort, created_at, status_code=200, latency_ms=100, actual=0.001, baseline=0.003):
            status = "applied" if cohort == "canary_applied" else "holdout"
            canary = {
                "enabled": True,
                "policy_id": "local-openai-canary-fixture",
                "rule_id": "local-openai-canary-fixture",
                "target_candidate_id": "openai-mini-fixture",
                "candidate_id": "openai-mini-fixture",
                "status": status,
                "cohort": cohort,
                "reason": "selected-canary" if status == "applied" else "selected-holdout",
                "original_model": "gpt-5-codex",
                "requested_model": "gpt-5-codex",
                "target_model": "gpt-5-mini",
                "actual_forwarded_model": "gpt-5-mini" if status == "applied" else "gpt-5-codex",
                "source_surface": "openai_provider_request",
                "app_family": "generic_openai",
                "category": "chat",
                "projected_input_savings_usd": 0.002,
                "canary_fraction": 0.5,
                "holdout_fraction": 0.25,
                "policy_source": "local-manual",
                "cohort_key_hash": f"sha256:{call_id}",
            }
            server.store.log_call(
                id=call_id,
                created_at=created_at,
                path="/v1/responses",
                requested_model="gpt-5-codex",
                routed_model=canary["actual_forwarded_model"],
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens_est=500,
                output_tokens_est=100,
                actual_input_tokens=500,
                actual_output_tokens=100,
                cost_est_usd=actual,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"openai_canary": canary}),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                error='{"message":"raw provider error must stay local"}' if status_code >= 400 else None,
                request_json=None,
                response_json=None,
                session_id="raw-openai-session-secret",
                category="chat",
                retry_count=0,
                provider="openai",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
            )

        fresh_base = datetime.now(timezone.utc) - timedelta(minutes=30)
        log_canary_call("openai-canary-ready-a1", cohort="canary_applied", created_at=fresh_base.isoformat())
        log_canary_call(
            "openai-canary-ready-a2",
            cohort="canary_applied",
            created_at=(fresh_base + timedelta(minutes=1)).isoformat(),
        )
        log_canary_call(
            "openai-canary-ready-h1",
            cohort="canary_holdout",
            created_at=(fresh_base + timedelta(minutes=2)).isoformat(),
            actual=0.003,
            baseline=0.003,
        )

        from agentflow_proxy import router

        with patch.dict(
            router.ROUTING_OPENAI_CANARY,
            {
                "enabled": True,
                "policy_id": "local-openai-canary-fixture",
                "target_candidate_id": "openai-mini-fixture",
                "target_model": "gpt-5-mini",
                "canary_fraction": 0.5,
                "holdout_fraction": 0.25,
                "policy_source": "local-manual",
            },
            clear=False,
        ):
            result = asyncio.run(stats_views.stats_openai_canary_readiness(server.store, limit=10))

        self.assertEqual(result["schema"], "agentflow.openai_canary_readiness.v1")
        self.assertEqual(result["state"], "ready_to_widen")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertEqual(result["policy"]["policy_id"], "local-openai-canary-fixture")
        self.assertEqual(result["summary"]["eligible_candidate_count"], 1)
        self.assertEqual(result["summary"]["canary_applied_count"], 2)
        self.assertEqual(result["summary"]["canary_holdout_count"], 1)
        self.assertEqual(result["summary"]["not_selected_count"], 0)
        self.assertEqual(result["summary"]["active_safety_stop_count"], 0)
        self.assertAlmostEqual(result["summary"]["observed_savings_usd"], 0.004, places=6)
        self.assertEqual(result["candidates"][0]["verdict"], "widen")
        self.assertEqual(result["candidates"][0]["applied_count"], 2)
        self.assertEqual(result["candidates"][0]["holdout_count"], 1)
        rendered = json.dumps(result)
        self.assertNotIn("raw-openai-session-secret", rendered)
        self.assertNotIn("raw provider error must stay local", rendered)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])

    def test_managed_pattern_adoption_funnel_covers_outcomes_and_lifecycle(self):
        pattern_hash = "sha256:" + ("a" * 64)

        def log_pattern_call(created_at, *, status_code=200, rules=None):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=10,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({
                    "changed": True,
                    "policy_source": "managed-recommended",
                    "pattern_rules": {
                        "configured_count": 1,
                        "policy_source": "managed-recommended",
                        "category": "tool-result",
                        "rules": rules or [],
                    },
                }),
                routing_json=stable_json({
                    "category": "tool-result",
                    "managed_recommendation": {
                        "enabled": True,
                        "status": "received",
                        "reason": "candidate matched",
                        "policy_source": "managed-recommended",
                        "policy_id": "candidate-crunch-pattern",
                        "applied": False,
                    },
                }),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="private-session-id",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        applied_rule = {
            "rule_id": "managed-crunch-pattern-candidate-crunch",
            "candidate_id": "candidate-crunch-pattern",
            "policy_source": "managed-recommended",
            "applied_count": 1,
            "saved_chars": 800,
            "matched_hashes": [pattern_hash],
            "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied", "fraction": 0.1},
        }
        holdout_rule = {
            **applied_rule,
            "applied_count": 0,
            "saved_chars": 0,
            "canary": {"enabled": True, "status": "holdout", "cohort": "canary_holdout", "fraction": 0.1},
        }
        bypass_rule = {
            **applied_rule,
            "applied_count": 0,
            "saved_chars": 0,
            "skip_reasons": [{"reason": "local-canary-safety-stop", "count": 1, "pattern_hash": pattern_hash}],
        }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AGENTFLOW_POLICY_EVENTS_LOG": os.path.join(tmp, "policy_events.jsonl")},
            clear=False,
        ):
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "fetch-review",
                ok=True,
                details={"source": "cli", "candidate_ids": ["candidate-crunch-pattern"], "candidate_count": 1},
            )
            log_policy_event(
                "apply",
                ok=True,
                details={"source": "cli", "dry_run": True, "applied_sections": ["crunch"], "changed_files": []},
            )
            log_policy_event(
                "rollout-actions-apply",
                ok=True,
                details={"source": "cli", "dry_run": False, "applied_sections": ["crunch"], "changed_action_count": 1},
            )
            log_policy_event(
                "rollback",
                ok=True,
                details={"source": "cli", "restored_sections": ["crunch"], "changed_files": ["crunch_rules.yaml"]},
            )
            log_policy_event(
                "rollout-actions-review",
                ok=False,
                details={"source": "cli", "error_type": "unsafe-rule-source", "action_count": 1},
            )

            log_pattern_call("2026-06-08T10:00:00+00:00", rules=[applied_rule])
            log_pattern_call("2026-06-08T10:01:00+00:00", rules=[holdout_rule])
            log_pattern_call("2026-06-08T10:02:00+00:00", rules=[bypass_rule])
            log_pattern_call("2026-06-08T10:03:00+00:00", status_code=500, rules=[applied_rule])

            result = asyncio.run(stats_views.stats_managed_recommendations(server.store, limit=20))

        adoption = result["adoption"]
        funnel = {row["stage"]: row["count"] for row in adoption["funnel"]}
        self.assertEqual(adoption["schema"], "agentflow.managed_pattern_adoption.v1")
        self.assertGreaterEqual(funnel["received"], 4)
        self.assertEqual(funnel["reviewed"], 1)
        self.assertEqual(funnel["dry_run"], 1)
        self.assertGreaterEqual(funnel["canary_applied"], 1)
        self.assertEqual(funnel["canary_holdout"], 1)
        self.assertEqual(funnel["bypassed"], 1)
        self.assertGreaterEqual(funnel["errored"], 1)
        self.assertEqual(funnel["rolled_back"], 1)
        self.assertEqual(funnel["rejected"], 2)
        self.assertFalse(adoption["privacy"]["raw_prompts_included"])
        self.assertFalse(adoption["privacy"]["tool_payloads_included"])
        self.assertFalse(adoption["privacy"]["local_session_ids_included"])

        outcome = next(row for row in adoption["pattern_outcomes_by_day"] if row["lifecycle_stage"] == "canary_applied")
        self.assertEqual(outcome["day"], "2026-06-08")
        self.assertEqual(outcome["policy_section"], "crunch")
        self.assertEqual(outcome["source_surface"], "anthropic_messages")
        self.assertEqual(outcome["app_family"], "claude_code")
        self.assertEqual(outcome["workflow_phase"], "tool-result")
        self.assertEqual(outcome["category"], "tool-result")
        self.assertEqual(outcome["policy_source"], "managed-recommended")
        self.assertEqual(outcome["candidate_id"], "candidate-crunch-pattern")
        self.assertEqual(outcome["rule_id"], "managed-crunch-pattern-candidate-crunch")
        self.assertEqual(outcome["pattern_hash"], pattern_hash)
        self.assertEqual(outcome["tokens_saved_est"], 200)

        blockers = {row["value"]: row["count"] for row in adoption["top_safety_blockers"]}
        self.assertEqual(blockers["local-canary-safety-stop"], 1)
        comparison = adoption["holdout_comparisons"][0]
        self.assertEqual(comparison["canary_applied_count"], 1)
        self.assertEqual(comparison["canary_holdout_count"], 1)
        self.assertEqual(comparison["bypassed_count"], 1)
        self.assertEqual(comparison["errored_count"], 1)
        lifecycle_stages = {row["lifecycle_stage"] for row in adoption["lifecycle_events"]}
        self.assertIn("reviewed", lifecycle_stages)
        self.assertIn("dry_run", lifecycle_stages)
        self.assertIn("applied", lifecycle_stages)
        json.dumps(result)

    def test_dashboard_managed_tab_renders_pattern_adoption_tables(self):
        html = stats_views.dashboard_html()

        self.assertIn("Managed pattern adoption funnel", html)
        self.assertIn("Managed pattern outcomes by day", html)
        self.assertIn("Managed pattern holdout comparison", html)
        self.assertIn("Managed pattern lifecycle events", html)
        self.assertIn("managed-pattern-funnel-tbody", html)
        self.assertIn("managed-pattern-outcomes-tbody", html)
        self.assertIn("managed-pattern-holdouts-tbody", html)
        self.assertIn("managed-pattern-lifecycle-tbody", html)
        self.assertIn("adoption.pattern_outcomes_by_day", html)

    def test_full_stats_unifies_source_surface_accounting_for_mixed_traffic(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=10,
            input_tokens_est=110,
            output_tokens_est=11,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 40, "policy_source": "local-default"}),
            routing_json=stable_json({"policy_source": "local-manual"}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="mixed-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=20,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5",
            routed_model="gpt-5-mini",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=20,
            input_tokens_est=210,
            output_tokens_est=21,
            actual_input_tokens=200,
            actual_output_tokens=20,
            cost_est_usd=0.002,
            cost_baseline_usd=0.004,
            crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
            routing_json=stable_json({"policy_source": "local-manual", "reason": "test route"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="mixed-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
        )
        start_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-mixed",
            thread_id="thread-mixed",
            message_chars=80,
            params_chars=10,
            input_items=1,
            input_text_chars=40,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="mixed-session",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-mixed",
            thread_id="thread-mixed",
            message_chars=20,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=20,
            error_code=None,
            error_message=None,
            latency_ms=200,
            session_id="mixed-session",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        accounting = result["executive_summary"]["accounting_today"]
        by_surface = {row["source_surface"]: row for row in accounting["source_surfaces"]}
        savings = {
            (row["source_surface"], row["optimization_type"]): row["savings_usd"]
            for row in result["today_savings_by_source_surface"]
        }

        self.assertEqual(set(by_surface), {"anthropic_messages", "openai_responses", "codex_turn"})
        self.assertEqual(by_surface["anthropic_messages"]["input_tokens"], 120)
        self.assertEqual(by_surface["openai_responses"]["input_tokens"], 200)
        self.assertEqual(by_surface["codex_turn"]["input_tokens"], 10)
        self.assertEqual(by_surface["anthropic_messages"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["openai_responses"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["codex_turn"]["token_basis"], "estimated-from-chars")
        self.assertEqual(accounting["input_tokens"], 330)
        self.assertEqual(accounting["output_tokens"], 35)
        self.assertEqual(accounting["total_tokens"], 365)
        self.assertEqual(result["executive_summary"]["tokens_today"]["total_tokens"], 365)
        self.assertGreater(savings[("anthropic_messages", "crunching")], 0)
        self.assertNotIn(("anthropic_messages", "cache"), savings)
        self.assertGreater(savings[("anthropic_messages", "provider_prompt_cache")], 0)
        self.assertGreater(savings[("openai_responses", "routing")], 0)
        self.assertNotIn(("codex_turn", "routing"), savings)
        json.dumps(result)

    def test_codex_effectiveness_report_summarizes_live_like_metadata_without_raw_text(self):
        secret = "secret prompt text"

        def log_turn(
            request_id,
            *,
            routing,
            crunch,
            cache,
            input_text_chars=0,
            params_chars=100,
            response_error_code=None,
            response_error_message=None,
            response_latency_ms=100,
        ):
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=f"thread-{request_id}",
                message_chars=200,
                params_chars=params_chars,
                input_items=1 if input_text_chars else 0,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-effectiveness",
                routing_json=stable_json(routing),
                crunch_json=stable_json(crunch),
                cache_json=stable_json(cache),
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=f"thread-{request_id}",
                message_chars=160,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=80,
                error_code=response_error_code,
                error_message=response_error_message,
                latency_ms=response_latency_ms,
                session_id="codex-effectiveness",
            )

        log_turn(
            "model-absent",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": False,
                    "status": "skipped",
                    "reason": "disabled",
                    "outcome_feedback": {"enabled": False, "status": "skipped", "reason": "disabled"},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True, "policy_source": "local-default"},
            input_text_chars=120,
        )
        log_turn(
            "model-routed",
            routing={
                "status": "applied",
                "reason": "small non-tool Sonnet request routed to Haiku",
                "applied": True,
                "model_field": "model",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": True,
                    "status": "received",
                    "policy_id": "codex-policy-1",
                    "target_model": "claude-haiku-4-5-20251001",
                    "applied": False,
                    "apply_reason": "codex-app-managed-recommendation-observed-only",
                    "outcome_feedback": {"enabled": True, "status": "sent", "reason": "accepted", "optimization_unit_id": 77},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True, "policy_source": "local-default"},
            input_text_chars=80,
            response_error_code=-32000,
            response_error_message="upstream rejected routed model",
            response_latency_ms=250,
        )
        log_turn(
            "action-like",
            routing={"status": "not-applied", "reason": "action-like-params", "applied": False, "policy_source": "local-default"},
            crunch={"status": "not-applied", "reason": "action-like-params", "applied": False},
            cache={"status": "skipped", "reason": "action-like-params", "eligible": False, "policy_source": "local-default"},
        )
        log_turn(
            "crunch-applied",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
            },
            crunch={
                "status": "applied",
                "reason": "codex-turn-start-crunched",
                "applied": True,
                "changed": True,
                "saved_chars": 1600,
                "tokens_saved_est": 400,
                "codex_repeated_scaffolding": {
                    "status": "applied",
                    "reason": "codex-repeated-scaffolding-crunched",
                    "saved_chars": 1200,
                    "patterns": [
                        {
                            "type": "repeated_input_section",
                            "count": 2,
                            "saved_chars_est": 700,
                            "hashes": ["abcdef123456"],
                        },
                        {
                            "type": "older_input_head_tail",
                            "count": 1,
                            "saved_chars_est": 500,
                            "hashes": ["123456abcdef"],
                        },
                    ],
                },
                "codex_patterns": [
                    {"type": "repeated_input_section", "count": 2, "saved_chars_est": 700},
                    {"type": "older_input_head_tail", "count": 1, "saved_chars_est": 500},
                ],
                "note": secret,
            },
            cache={"status": "skipped", "reason": "unknown-param-shape", "eligible": False, "policy_source": "local-default"},
            input_text_chars=3000,
        )
        log_turn(
            "cache-hit",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": True,
                    "status": "error",
                    "reason": "server-error",
                    "outcome_feedback": {"enabled": True, "status": "error", "reason": "request-failed"},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "hit", "reason": "exact-match", "eligible": True, "hit_type": "exact", "policy_source": "local-default"},
            input_text_chars=200,
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=20))
        summary = result["summary"]

        self.assertEqual(result["schema"], "agentflow.codex_app_effectiveness.v1")
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_params_included"])
        self.assertFalse(result["privacy"]["raw_responses_included"])
        self.assertEqual(summary["turn_start_rows"], 5)
        self.assertEqual(summary["model_field_present"], 1)
        self.assertEqual(summary["model_field_absent"], 3)
        self.assertEqual(summary["routing_applied"], 1)
        self.assertEqual(summary["crunch_applied"], 1)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["cache_eligible"], 3)
        self.assertEqual(summary["action_like_skips"], 1)
        self.assertEqual(summary["unknown_param_skips"], 1)
        self.assertEqual(summary["total_saved_chars"], 1600)
        self.assertEqual(summary["total_saved_tokens_est"], 400)
        self.assertEqual(summary["codex_repeated_scaffolding_saved_chars"], 1200)
        self.assertEqual(summary["error_rows"], 1)
        self.assertEqual(summary["optimized_rows"], 3)
        self.assertEqual(summary["pass_through_rows"], 2)
        self.assertGreater(summary["optimized_error_rate"], 0)
        self.assertEqual(summary["managed_recommendation_rows"], 5)
        self.assertEqual(summary["managed_recommendation_enabled"], 2)
        self.assertEqual(summary["managed_recommendation_disabled"], 3)
        self.assertEqual(summary["managed_feedback_sent"], 1)
        self.assertEqual(summary["managed_feedback_skipped"], 1)
        self.assertEqual(summary["managed_feedback_error"], 1)

        model_fields = {row["value"]: row["count"] for row in result["model_field_breakdown"]}
        self.assertEqual(model_fields["present"], 1)
        self.assertEqual(model_fields["absent"], 3)
        shapes = {row["value"]: row["count"] for row in result["param_shape_breakdown"]}
        self.assertEqual(shapes["action-like-params"], 1)
        self.assertEqual(shapes["unknown-param-shape"], 1)
        routing_statuses = {row["status"] for row in result["routing_breakdown"]}
        self.assertIn("applied", routing_statuses)
        self.assertIn("not-applicable", routing_statuses)
        patterns = {row["type"]: row for row in result["crunch_pattern_breakdown"]}
        self.assertEqual(patterns["repeated_input_section"]["count"], 2)
        self.assertEqual(patterns["older_input_head_tail"]["saved_chars_est"], 500)
        feedback_statuses = {row["value"]: row["count"] for row in result["managed_feedback_breakdown"]}
        self.assertEqual(feedback_statuses, {"sent": 1, "skipped": 1, "error": 1, "pending": 2})
        sample = next(row for row in result["recent_samples"] if row["saved_chars"] == 1600)
        self.assertEqual(set(sample["codex_pattern_types"]), {"repeated_input_section", "older_input_head_tail"})
        self.assertNotIn(secret, json.dumps(result))

    def test_codex_effectiveness_reports_summary_model_hint_canary_buckets_by_phase(self):
        forbidden_secret = "raw summary prompt must not appear"

        def log_hint_turn(
            request_id,
            *,
            routing,
            crunch=None,
            cache=None,
            input_text_chars=4000,
            result_chars=400,
            response_error_code=None,
            response_latency_ms=100,
            with_response=True,
        ):
            server.store.log_codex_app_event(
                id=f"start-hint-{request_id}",
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=f"thread-hint-{request_id}",
                message_chars=input_text_chars + 100,
                params_chars=input_text_chars + 50,
                input_items=1,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-summary-hint",
                routing_json=stable_json(routing),
                crunch_json=stable_json(crunch or {"status": "skipped", "reason": "no-change", "applied": False}),
                cache_json=stable_json(cache or {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False}),
            )
            if with_response:
                server.store.log_codex_app_event(
                    id=f"response-hint-{request_id}",
                    created_at=utc_now(),
                    direction="server_to_client",
                    method="turn/completed",
                    request_id=request_id,
                    thread_id=f"thread-hint-{request_id}",
                    message_chars=result_chars + 80,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=result_chars,
                    error_code=response_error_code,
                    error_message="safe jsonrpc error" if response_error_code is not None else None,
                    latency_ms=response_latency_ms,
                    session_id="codex-summary-hint",
                )

        log_hint_turn(
            "applied",
            routing={
                "status": "applied",
                "reason": "safe-summary-model-hint-canary",
                "applied": True,
                "canary": "codex-app-summary-model-hint",
                "canary_enabled": True,
                "model_field": "model",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5-codex",
                "target_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "summary_model_hint": {
                    "status": "applied",
                    "eligible": True,
                    "requested_model": "gpt-5.3-codex",
                    "target_model": "gpt-5-codex",
                    "model_field_state": "present",
                    "workflow_phase": "summary",
                    "workflow_phase_reason": "summary-text-intent",
                    "estimated_cost_delta": {
                        "basis": "input-text-chars-estimated",
                        "input_tokens_est": 1000,
                        "delta_usd": 0.0005,
                        "cost_known": True,
                    },
                },
            },
            crunch={
                "status": "applied",
                "reason": "codex-repeated-scaffolding-crunched",
                "applied": True,
                "changed": True,
                "saved_chars": 320,
                "tokens_saved_est": 80,
                "note": forbidden_secret,
            },
            cache={"status": "miss", "reason": "exact-miss", "eligible": True, "workflow_phase": "summary"},
        )
        log_hint_turn(
            "eligible",
            routing={
                "status": "skipped",
                "reason": "summary-model-hint-target-matches-requested",
                "applied": False,
                "canary": "codex-app-summary-model-hint",
                "canary_enabled": True,
                "model_field": "model",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "target_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "summary_model_hint": {
                    "status": "eligible-skipped",
                    "eligible": True,
                    "skip_reason": "summary-model-hint-target-matches-requested",
                    "requested_model": "gpt-5-codex",
                    "target_model": "gpt-5-codex",
                    "model_field_state": "present",
                    "workflow_phase": "summary",
                    "estimated_cost_delta": {"delta_usd": 0.0, "cost_known": True},
                },
            },
            cache={"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True, "workflow_phase": "summary"},
            with_response=False,
        )
        log_hint_turn(
            "holdout",
            routing={
                "status": "skipped",
                "reason": "summary-model-hint-canary-holdout",
                "applied": False,
                "canary": "codex-app-summary-model-hint",
                "canary_enabled": True,
                "canary_cohort": "canary_holdout",
                "model_field": "model",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.3-codex",
                "target_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "summary_model_hint": {
                    "status": "holdout",
                    "eligible": True,
                    "skip_reason": "summary-model-hint-canary-holdout",
                    "canary_cohort": "canary_holdout",
                    "requested_model": "gpt-5.3-codex",
                    "target_model": "gpt-5-codex",
                    "model_field_state": "present",
                    "workflow_phase": "summary",
                    "estimated_cost_delta": {"delta_usd": 0.0004, "cost_known": True},
                },
            },
            cache={"status": "miss", "reason": "exact-miss", "eligible": True, "workflow_phase": "summary"},
            response_latency_ms=180,
        )
        log_hint_turn(
            "unsafe",
            routing={
                "status": "skipped",
                "reason": "action-like-params",
                "applied": False,
                "canary": "codex-app-summary-model-hint",
                "canary_enabled": True,
                "model_field": "model",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.3-codex",
                "target_model": "gpt-5-codex",
                "workflow_phase": "tool_execution",
                "summary_model_hint": {
                    "status": "unsafe-skipped",
                    "eligible": False,
                    "skip_reason": "action-like-params",
                    "requested_model": "gpt-5.3-codex",
                    "target_model": "gpt-5-codex",
                    "model_field_state": "present",
                    "workflow_phase": "tool_execution",
                },
            },
            cache={"status": "skipped", "reason": "action-like-params", "eligible": False, "workflow_phase": "tool_execution"},
            response_error_code=-32000,
            response_latency_ms=250,
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        hint = result["summary_model_hint"]
        by_key = {(row["workflow_phase"], row["status"]): row for row in hint["buckets"]}

        self.assertEqual(hint["summary"]["turns"], 4)
        self.assertEqual(hint["summary"]["applied"], 1)
        self.assertEqual(hint["summary"]["holdout"], 1)
        self.assertEqual(hint["summary"]["eligible_skipped"], 1)
        self.assertEqual(hint["summary"]["unsafe_skipped"], 1)
        self.assertEqual(hint["summary"]["pending"], 1)
        self.assertEqual(hint["summary"]["errors"], 1)
        self.assertGreater(hint["summary"]["estimated_savings_usd"], 0)
        self.assertEqual(hint["summary"]["candidate_count"], 3)
        self.assertEqual(hint["summary"]["unsafe_skip_count"], 1)
        self.assertEqual(hint["canary"]["candidate_count"], 3)
        self.assertEqual(hint["canary"]["applied_count"], 1)
        self.assertEqual(hint["canary"]["holdout_count"], 1)
        self.assertEqual(hint["canary"]["unsafe_skip_count"], 1)
        self.assertGreater(hint["canary"]["candidate_projected_savings_usd"], 0)
        self.assertGreater(hint["canary"]["holdout_projected_savings_usd"], 0)
        self.assertEqual(hint["canary"]["applied_minus_holdout_error_rate"], 0)
        self.assertEqual(hint["canary"]["applied_minus_holdout_latency_avg_ms"], -80)
        self.assertGreater(by_key[("summary", "applied")]["estimated_savings_usd"], 0)
        self.assertGreater(by_key[("summary", "holdout")]["projected_savings_usd"], 0)
        self.assertEqual(by_key[("summary", "applied")]["crunch_applied"], 1)
        self.assertEqual(by_key[("summary", "applied")]["cache_eligible"], 1)
        self.assertEqual(by_key[("summary", "eligible-skipped")]["pending"], 1)
        self.assertEqual(by_key[("tool_execution", "unsafe-skipped")]["errors"], 1)
        self.assertEqual(by_key[("tool_execution", "unsafe-skipped")]["error_rate"], 1.0)
        self.assertEqual(by_key[("tool_execution", "unsafe-skipped")]["avg_latency_ms"], 250)
        self.assertEqual(result["summary"]["summary_model_hint_applied"], 1)
        self.assertEqual(result["summary"]["summary_model_hint_holdout"], 1)
        self.assertEqual(result["summary"]["summary_model_hint_eligible_skipped"], 1)
        self.assertEqual(result["summary"]["summary_model_hint_unsafe_skipped"], 1)
        self.assertFalse(hint["privacy"]["raw_params_included"])
        self.assertFalse(hint["privacy"]["raw_request_ids_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/codex-effectiveness?limit=10")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary_model_hint"]["summary"]["turns"], 4)
        html = stats_views.dashboard_html()
        self.assertIn("<h2>Summary model hint canary</h2>", html)
        self.assertIn("id=\"codex-summary-hint-tbody\"", html)
        self.assertIn("summary_model_hint", html)

        rendered = json.dumps(result)
        forbidden = {"prompt", "messages", "content", "raw_request", "raw_response", "params", "transcript", "input"}
        self.assertTrue(forbidden.isdisjoint(self._keys_in(result)))
        self.assertNotIn(forbidden_secret, rendered)

    def test_codex_readiness_reports_canary_cache_policy_and_privacy(self):
        secret = "raw codex readiness payload must not appear"

        def log_turn(
            request_id,
            *,
            routing,
            cache,
            thread_id,
            input_text_chars=400,
            result_chars=80,
        ):
            server.store.log_codex_app_event(
                id=f"start-readiness-{request_id}",
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=input_text_chars + 10,
                params_chars=input_text_chars + 5,
                input_items=1,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-readiness-session-secret",
                routing_json=stable_json(routing),
                crunch_json=stable_json({"status": "skipped", "reason": "no-change", "changed": False, "debug": secret}),
                cache_json=stable_json(cache),
            )
            server.store.log_codex_app_event(
                id=f"response-readiness-{request_id}",
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=result_chars + 10,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=None,
                error_message=None,
                latency_ms=100,
                session_id="codex-readiness-session-secret",
            )

        log_turn(
            "applied",
            thread_id="thread-readiness-applied",
            routing={
                "status": "applied",
                "reason": "safe-summary-model-hint-canary",
                "applied": True,
                "policy_source": "local-manual",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "summary_model_hint": {
                    "status": "applied",
                    "eligible": True,
                    "requested_model": "gpt-5.3-codex",
                    "target_model": "gpt-5-codex",
                    "workflow_phase": "summary",
                    "estimated_cost_delta": {"delta_usd": 0.0001, "cost_known": True},
                },
            },
            cache={
                "status": "hit",
                "reason": "exact-match",
                "hit_type": "exact",
                "eligible": True,
                "policy_source": "local-manual",
                "surface": "codex_turn",
                "cache_key": "secret-cache-key-must-not-appear",
            },
        )
        log_turn(
            "holdout",
            thread_id="thread-readiness-holdout",
            routing={
                "status": "skipped",
                "reason": "summary-model-hint-canary-holdout",
                "applied": False,
                "policy_source": "local-manual",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.3-codex",
                "workflow_phase": "summary",
                "summary_model_hint": {
                    "status": "holdout",
                    "eligible": True,
                    "canary_cohort": "canary_holdout",
                    "requested_model": "gpt-5.3-codex",
                    "target_model": "gpt-5-codex",
                    "workflow_phase": "summary",
                    "estimated_cost_delta": {"delta_usd": 0.0002, "cost_known": True},
                },
            },
            cache={
                "status": "holdout",
                "reason": "codex-app-cache-canary-holdout",
                "eligible": True,
                "policy_source": "local-manual",
                "surface": "codex_turn",
            },
        )
        log_turn(
            "miss",
            thread_id="thread-readiness-miss",
            routing={"status": "skipped", "reason": "not-summary", "workflow_phase": "tool_execution"},
            cache={"status": "miss", "reason": "exact-miss", "eligible": True, "policy_source": "local-manual", "surface": "codex_turn"},
        )
        log_turn(
            "invalidated",
            thread_id="thread-readiness-invalidated",
            routing={"status": "skipped", "reason": "not-summary", "workflow_phase": "verification"},
            cache={"status": "miss", "reason": "dependency-changed", "eligible": True, "policy_source": "local-manual", "surface": "codex_turn"},
        )
        server.store.log_codex_app_event(
            id="token-readiness-applied",
            created_at=utc_now(),
            direction="server_to_client",
            method="thread/tokenUsage/updated",
            request_id=None,
            thread_id="thread-readiness-applied",
            message_chars=120,
            params_chars=120,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-readiness-session-secret",
            metadata_json=stable_json({
                "schema": "agentflow.codex_app_metadata.v1",
                "kind": "token_usage",
                "method": "thread/tokenUsage/updated",
                "token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 120,
                },
                "raw_payload": secret,
            }),
        )

        result = asyncio.run(stats_views.stats_codex_readiness(server.store, limit=20))

        self.assertEqual(result["schema"], "agentflow.codex_optimization_readiness.v1")
        self.assertEqual(result["source_surface"], "codex_turn")
        self.assertEqual(result["summary"]["turn_start_rows"], 4)
        self.assertEqual(result["summary"]["phase_known_rate"], 1.0)
        self.assertEqual(result["summary"]["token_reconciliation_status"], "reconciled")
        self.assertEqual(result["summary"]["summary_model_hint_applied"], 1)
        self.assertEqual(result["summary"]["summary_model_hint_holdout"], 1)
        self.assertEqual(result["summary"]["exact_cache_hits"], 1)
        self.assertEqual(result["summary"]["exact_cache_holdouts"], 1)
        self.assertEqual(result["summary"]["exact_cache_misses"], 1)
        self.assertEqual(result["summary"]["exact_cache_invalidations"], 1)
        self.assertGreater(result["summary"]["summary_model_hint_estimated_savings_usd"], 0)
        self.assertGreater(result["exact_cache"]["estimated_saved_cost_usd"], 0)
        self.assertIn("policy_source", result["policy"])
        self.assertIn("rule_path", result["policy"])
        self.assertIn("readiness_checks", result)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_params_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["thread_ids_included"])
        self.assertFalse(result["privacy"]["local_session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])

        cohorts = {row["cohort"]: row["count"] for row in result["exact_cache"]["cohorts"]}
        self.assertEqual(cohorts["hit"], 1)
        self.assertEqual(cohorts["holdout"], 1)
        self.assertEqual(cohorts["miss"], 1)
        self.assertEqual(cohorts["invalidated"], 1)

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/codex-readiness?limit=20")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["turn_start_rows"], 4)
        html = stats_views.dashboard_html()
        self.assertIn("/agentflow/stats/codex-readiness", html)
        self.assertIn("Codex optimization readiness", html)
        self.assertIn("codex-readiness-tbody", html)
        self.assertIn("codex-cache-readiness-tbody", html)

        rendered = json.dumps(result)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("secret-cache-key-must-not-appear", rendered)
        self.assertNotIn("codex-readiness-session-secret", rendered)

    def test_codex_effectiveness_endpoint_keeps_terminal_compaction_metadata_content_free(self):
        raw_request_id = "raw-codex-terminal-dashboard-request-id-must-not-leak"
        raw_thread_id = "raw-codex-terminal-dashboard-thread-id-must-not-leak"
        raw_session_id = "raw-codex-terminal-dashboard-session-id-must-not-leak"
        raw_terminal = "raw codex terminal dashboard transcript must not leak"
        raw_path = "/workspace/private/raw-codex-terminal-dashboard.log"
        raw_cache_key = "raw-codex-terminal-dashboard-cache-key-must-not-leak"
        terminal_family = "codex_terminal_transcript_compaction"

        server.store.log_codex_app_event(
            id="start-codex-terminal-dashboard",
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id=raw_request_id,
            thread_id=raw_thread_id,
            message_chars=42_000,
            params_chars=41_000,
            input_items=1,
            input_text_chars=40_000,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id=raw_session_id,
            routing_json=stable_json({
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "workflow_phase": "tool_execution",
            }),
            crunch_json=stable_json({
                "status": "applied",
                "reason": "terminal-transcript-compacted",
                "applied": True,
                "changed": True,
                "saved_chars": 8000,
                "tokens_saved_est": 2000,
                terminal_family: {
                    "status": "applied",
                    "reason": "terminal-transcript-compacted",
                    "candidate_id": raw_path,
                    "rule_id": "raw-dashboard-terminal-rule-id-must-not-leak",
                    "debug": raw_terminal,
                    "provider_body": {"raw": raw_terminal},
                    "raw_text_included": False,
                    "raw_commands_included": False,
                },
            }),
            cache_json=stable_json({
                "status": "skipped",
                "reason": "codex-app-cache-disabled",
                "eligible": False,
                "cache_key": raw_cache_key,
            }),
            event_window_json=stable_json({
                "schema": "agentflow.codex_app_event_window.v1",
                "workflow_phase": "tool_execution",
                "input_text_chars": 40_000,
                "method_counts": {"turn/start": 1, "item/commandExecution/outputDelta": 10},
                "request_id": raw_request_id,
                "thread_id": raw_thread_id,
                "session_id": raw_session_id,
                "file_path": raw_path,
                "provider_body": {"raw": raw_terminal},
            }),
        )
        server.store.log_codex_app_event(
            id="response-codex-terminal-dashboard",
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id=raw_request_id,
            thread_id=raw_thread_id,
            message_chars=200,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=200,
            error_code=None,
            error_message=raw_terminal,
            latency_ms=120,
            session_id=raw_session_id,
        )

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/codex-effectiveness?limit=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["summary"]["turn_start_rows"], 1)
        self.assertEqual(payload["summary"]["crunch_applied"], 1)
        self.assertEqual(payload["summary"]["total_saved_chars"], 8000)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["raw_params_included"])
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            raw_request_id,
            raw_thread_id,
            raw_session_id,
            raw_terminal,
            raw_path,
            raw_cache_key,
            "raw-dashboard-terminal-rule-id-must-not-leak",
            "/workspace/private",
            '"provider_body"',
            '"cache_key"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_codex_effectiveness_reports_quota_token_usage_without_raw_payloads(self):
        raw_prompt = "seeded raw prompt must not appear"
        raw_command = "seeded raw command must not appear"
        raw_transcript = "seeded raw transcript must not appear"
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="quota-turn",
            thread_id="thread-quota",
            message_chars=200,
            params_chars=100,
            input_items=1,
            input_text_chars=4000,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            routing_json=stable_json({"status": "not-applicable", "reason": "codex-turn-start-model-field-absent"}),
            crunch_json=stable_json({"status": "skipped", "changed": False}),
            cache_json=stable_json({"status": "skipped", "eligible": False}),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="quota-turn",
            thread_id="thread-quota",
            message_chars=80,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=400,
            error_code=None,
            error_message=None,
            latency_ms=50,
            session_id="codex-quota",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="thread/tokenUsage/updated",
            request_id=None,
            thread_id="thread-quota",
            message_chars=120,
            params_chars=120,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            metadata_json=stable_json({
                "schema": "agentflow.codex_app_metadata.v1",
                "kind": "token_usage",
                "method": "thread/tokenUsage/updated",
                "token_usage": {
                    "input_tokens": 1200,
                    "cached_input_tokens": 200,
                    "output_tokens": 125,
                    "reasoning_output_tokens": 25,
                    "total_tokens": 1550,
                    "total_tokens_bucket": "1k_10k",
                },
                "debug": raw_prompt,
            }),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="account/rateLimits/updated",
            request_id=None,
            thread_id=None,
            message_chars=120,
            params_chars=120,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            metadata_json=stable_json({
                "schema": "agentflow.codex_app_metadata.v1",
                "kind": "rate_limits",
                "method": "account/rateLimits/updated",
                "rate_limits": {
                    "plan_type": "pro",
                    "pressure": "high",
                    "scopes": [
                        {
                            "name": "primary",
                            "used_percent": 92.5,
                            "used_percent_bucket": "90_99",
                            "remaining": 42,
                            "remaining_bucket": "10_99",
                            "reset_bucket": "1m_1h",
                        }
                    ],
                },
                "debug": raw_command,
                "transcript": raw_transcript,
            }),
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        quota = result["quota_and_token_usage"]

        self.assertEqual(result["summary"]["rate_limit_update_rows"], 1)
        self.assertEqual(result["summary"]["token_usage_update_rows"], 1)
        self.assertEqual(quota["latest_rate_limits"]["pressure"], "high")
        self.assertEqual(quota["latest_rate_limits"]["scopes"][0]["remaining_bucket"], "10_99")
        self.assertEqual(quota["token_usage_totals"]["total_tokens"], 1550)
        self.assertEqual(quota["agentflow_estimated_totals"]["total_tokens_est"], 1100)
        self.assertEqual(quota["reconciliation"]["total_drift_tokens"], 450)
        self.assertEqual(quota["reconciliation"]["total_drift_bucket"], "reconciled")
        self.assertEqual(quota["reconciliation"]["total_drift_size_bucket"], "100_999")
        self.assertEqual(quota["matched_agentflow_estimated_totals"]["total_tokens_est"], 1100)
        self.assertEqual(quota["latest_token_usage_delta"]["total_tokens"], 1550)
        self.assertGreater(quota["reconciled_cost_usd"], 0)
        self.assertEqual(quota["by_workflow_phase"][0]["workflow_phase"], "unknown")
        self.assertEqual(quota["by_workflow_phase"][0]["total_tokens"], 1550)
        self.assertEqual(quota["by_model"][0]["processing_mode"], stats_views.CODEX_APP_PROCESSING_MODE)
        self.assertEqual(quota["by_thread"][0]["scope_type"], "thread")
        self.assertTrue(quota["by_thread"][0]["scope_hash"].startswith("sha256:"))
        self.assertFalse(quota["by_thread"][0]["thread_id_included"])
        self.assertFalse(quota["privacy"]["raw_prompts_included"])
        self.assertFalse(quota["privacy"]["raw_commands_included"])
        self.assertFalse(quota["privacy"]["raw_thread_ids_included"])
        rendered = json.dumps(result)
        self.assertNotIn(raw_prompt, rendered)
        self.assertNotIn(raw_command, rendered)
        self.assertNotIn(raw_transcript, rendered)
        self.assertNotIn("thread-quota", rendered)
        self.assertNotIn("codex-quota", rendered)

    def test_codex_effectiveness_reconciles_token_usage_deltas_and_reasons(self):
        def log_turn(request_id, thread_id, created_at, *, session_id="codex-deltas", phase="summary"):
            server.store.log_codex_app_event(
                id=f"start-{request_id}",
                created_at=created_at,
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=200,
                params_chars=100,
                input_items=1,
                input_text_chars=400,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=session_id,
                routing_json=stable_json({"status": "not-applied", "workflow_phase": phase}),
                crunch_json=stable_json({"status": "skipped", "changed": False, "workflow_phase": phase}),
                cache_json=stable_json({"status": "skipped", "eligible": False, "workflow_phase": phase}),
            )
            server.store.log_codex_app_event(
                id=f"response-{request_id}",
                created_at=created_at.replace(":00+00:00", ":30+00:00"),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=80,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=120,
                error_code=None,
                error_message=None,
                latency_ms=50,
                session_id=session_id,
            )

        def log_usage(
            event_id,
            created_at,
            usage,
            *,
            thread_id=None,
            session_id="codex-deltas",
        ):
            server.store.log_codex_app_event(
                id=event_id,
                created_at=created_at,
                direction="server_to_client",
                method="thread/tokenUsage/updated",
                request_id=None,
                thread_id=thread_id,
                message_chars=80,
                params_chars=80,
                input_items=None,
                input_text_chars=None,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=session_id,
                metadata_json=stable_json({
                    "schema": "agentflow.codex_app_metadata.v1",
                    "kind": "token_usage",
                    "method": "thread/tokenUsage/updated",
                    "token_usage": {
                        **usage,
                        "total_tokens": sum(usage.values()),
                        "total_tokens_bucket": "1_9",
                    },
                }),
            )

        log_turn("one", "thread-delta", "2026-06-10T12:00:00+00:00", phase="summary")
        log_turn("two", "thread-delta", "2026-06-10T12:02:00+00:00", phase="tool_execution")
        log_usage(
            "usage-one",
            "2026-06-10T12:00:45+00:00",
            {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 30, "reasoning_output_tokens": 0},
            thread_id="thread-delta",
        )
        log_usage(
            "usage-two",
            "2026-06-10T12:02:45+00:00",
            {"input_tokens": 160, "cached_input_tokens": 25, "output_tokens": 55, "reasoning_output_tokens": 0},
            thread_id="thread-delta",
        )
        log_usage(
            "usage-reset",
            "2026-06-10T12:03:15+00:00",
            {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5, "reasoning_output_tokens": 0},
            thread_id="thread-delta",
        )
        log_usage(
            "usage-aggregate",
            "2026-06-10T12:03:30+00:00",
            {"input_tokens": 7, "cached_input_tokens": 0, "output_tokens": 3, "reasoning_output_tokens": 0},
            thread_id=None,
            session_id="codex-deltas",
        )
        log_usage(
            "usage-missing",
            "2026-06-10T12:04:00+00:00",
            {"input_tokens": 5, "cached_input_tokens": 0, "output_tokens": 1, "reasoning_output_tokens": 0},
            thread_id=None,
            session_id=None,
        )
        log_usage(
            "usage-stale",
            "2026-06-10T12:05:00+00:00",
            {"input_tokens": 8, "cached_input_tokens": 0, "output_tokens": 2, "reasoning_output_tokens": 0},
            thread_id="thread-stale",
            session_id="codex-deltas",
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        quota = result["quota_and_token_usage"]
        statuses = {row["value"]: row["count"] for row in quota["reconciliation"]["status_breakdown"]}
        status_tokens = {row["status"]: row["total_tokens"] for row in quota["reconciliation"]["status_token_totals"]}

        self.assertEqual(statuses["reconciled"], 2)
        self.assertEqual(statuses["reset"], 1)
        self.assertEqual(statuses["aggregate-only"], 1)
        self.assertEqual(statuses["missing-thread"], 1)
        self.assertEqual(statuses["stale"], 1)
        self.assertEqual(status_tokens["reconciled"], 240)
        self.assertEqual(status_tokens["reset"], 17)
        self.assertEqual(status_tokens["aggregate-only"], 10)
        self.assertEqual(status_tokens["missing-thread"], 6)
        self.assertEqual(status_tokens["stale"], 10)
        self.assertEqual(quota["token_usage_totals"]["total_tokens"], 283)
        self.assertEqual(quota["raw_counter_totals"]["total_tokens"], 433)
        self.assertEqual(quota["matched_agentflow_estimated_totals"]["total_tokens_est"], 260)
        self.assertEqual(quota["reconciliation"]["total_drift_bucket"], "reset")
        self.assertEqual(result["summary"]["token_usage_reconciliation_drift_bucket"], "reset")
        phase_tokens = {row["workflow_phase"]: row["total_tokens"] for row in quota["by_workflow_phase"]}
        self.assertEqual(phase_tokens["summary"], 150)
        self.assertEqual(phase_tokens["tool_execution"], 90)
        self.assertEqual(phase_tokens["reset"], 17)
        self.assertEqual(phase_tokens["aggregate-only"], 10)
        self.assertEqual(phase_tokens["missing-thread"], 6)
        self.assertEqual(phase_tokens["stale"], 10)
        self.assertTrue(all(row["scope_hash"].startswith("sha256:") for row in quota["by_thread"]))
        rendered = json.dumps(result)
        self.assertNotIn("thread-delta", rendered)
        self.assertNotIn("thread-stale", rendered)
        self.assertNotIn("codex-deltas", rendered)

    def test_codex_effectiveness_normalizes_historical_missing_decision_metadata(self):
        fixtures = [
            (
                "complete",
                stable_json({"status": "not-applied", "reason": "fixture-route", "applied": False, "policy_source": "local-default"}),
                stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
                None,
            ),
            ("historical", None, None, None, None),
            (
                "partial",
                None,
                stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
                None,
            ),
            (
                "current-missing",
                None,
                None,
                None,
                stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 1,
                    "method_counts": {"turn/start": 1},
                    "direction_counts": {"client_to_server": 1},
                    "model_field_state": "derived_absent",
                }),
            ),
        ]
        for index, (suffix, routing_json, crunch_json, cache_json, event_window_json) in enumerate(fixtures):
            server.store.log_codex_app_event(
                id=f"start-decision-{suffix}",
                created_at=f"2026-06-08T11:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-decision-{suffix}",
                thread_id=f"thread-decision-{suffix}",
                message_chars=160,
                params_chars=90,
                input_items=1,
                input_text_chars=72,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-decision-metadata",
                routing_json=routing_json,
                crunch_json=crunch_json,
                cache_json=cache_json,
                event_window_json=event_window_json,
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        summary = result["summary"]
        metadata = {row["value"]: row["count"] for row in result["decision_metadata_breakdown"]}
        routing: dict[str, int] = {}
        for row in result["routing_breakdown"]:
            routing[row["status"]] = routing.get(row["status"], 0) + row["count"]
        current_missing = {row["value"]: row["count"] for row in result["current_missing_decision_breakdown"]}
        not_instrumented = {row["value"]: row["count"] for row in result["not_instrumented_decision_breakdown"]}
        historical = {row["value"]: row["count"] for row in result["historical_unavailable_decision_breakdown"]}
        sample_states = {row["decision_metadata_state"] for row in result["recent_samples"]}

        self.assertEqual(summary["turn_start_rows"], 4)
        self.assertEqual(summary["decision_metadata_complete_rows"], 2)
        self.assertEqual(summary["decision_metadata_historical_unavailable_rows"], 0)
        self.assertEqual(summary["decision_metadata_not_instrumented_rows"], 2)
        self.assertEqual(summary["decision_metadata_current_missing_rows"], 0)
        self.assertEqual(summary["current_missing_decisions"], 0)
        self.assertEqual(summary["not_instrumented_decisions"], 4)
        self.assertEqual(summary["historical_unavailable_decisions"], 0)
        self.assertEqual(metadata, {
            "complete": 2,
            "not-instrumented": 2,
        })
        self.assertEqual(routing["not-applied"], 4)
        self.assertEqual(current_missing, {})
        self.assertEqual(not_instrumented, {"crunch": 2, "cache": 2})
        self.assertEqual(historical, {})
        self.assertIn("not-instrumented", sample_states)
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_codex_effectiveness_classifies_workflow_phases_from_event_sequences(self):
        def log_turn(
            name,
            *,
            start_at,
            signal_method=None,
            phase_thread=None,
            routing=None,
            crunch=None,
            cache=None,
            input_text_chars=120,
            result_chars=80,
            response_error_code=None,
            latency_ms=100,
        ):
            thread_id = phase_thread or f"thread-{name}"
            request_id = f"req-{name}"
            server.store.log_codex_app_event(
                id=f"start-{name}",
                created_at=f"2026-06-08T10:{start_at:02d}:00+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=200,
                params_chars=100,
                input_items=1 if input_text_chars else 0,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-phase-session",
                routing_json=stable_json(routing or {
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                    "policy_source": "local-default",
                }),
                crunch_json=stable_json(crunch or {"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                cache_json=stable_json(cache or {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
            )
            if signal_method:
                server.store.log_codex_app_event(
                    id=f"signal-{name}",
                    created_at=f"2026-06-08T10:{start_at:02d}:01+00:00",
                    direction="server_to_client",
                    method=signal_method,
                    request_id=None,
                    thread_id=thread_id,
                    message_chars=90,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-phase-session",
                )
            server.store.log_codex_app_event(
                id=f"end-{name}",
                created_at=f"2026-06-08T10:{start_at:02d}:02+00:00",
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=120,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=response_error_code,
                error_message="phase fixture error" if response_error_code is not None else None,
                latency_ms=latency_ms,
                session_id="codex-phase-session",
            )

        log_turn(
            "planning",
            start_at=0,
            signal_method="turn/plan/updated",
            routing={"status": "applied", "reason": "fixture-route", "applied": True, "policy_source": "local-default"},
            input_text_chars=400,
        )
        log_turn(
            "tool",
            start_at=1,
            signal_method="item/commandExecution/outputDelta",
            response_error_code=-32000,
            latency_ms=300,
            input_text_chars=240,
        )
        log_turn(
            "verification",
            start_at=2,
            signal_method="turn/diff/updated",
            crunch={"status": "applied", "reason": "fixture-crunch", "applied": True, "changed": True, "saved_chars": 40, "tokens_saved_est": 10},
            input_text_chars=160,
        )
        log_turn(
            "summary",
            start_at=3,
            signal_method="item/agentMessage/delta",
            cache={"status": "hit", "reason": "exact-match", "eligible": True, "hit_type": "exact", "policy_source": "local-default"},
            input_text_chars=80,
        )
        log_turn("idle", start_at=4, input_text_chars=0, result_chars=10)
        log_turn("unknown", start_at=5, input_text_chars=140)

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=20))
        phases = {row["phase"]: row for row in result["workflow_phase_breakdown"]}

        self.assertEqual(result["summary"]["turn_start_rows"], 6)
        self.assertEqual(result["summary"]["workflow_phase_known"], 5)
        self.assertEqual(result["summary"]["workflow_phase_unknown"], 1)
        self.assertEqual(phases["planning"]["routing_applied"], 1)
        self.assertEqual(phases["tool_execution"]["errors"], 1)
        self.assertEqual(phases["tool_execution"]["avg_latency_ms"], 300)
        self.assertEqual(phases["verification"]["crunch_applied"], 1)
        self.assertEqual(phases["verification"]["saved_chars"], 40)
        self.assertEqual(phases["summary"]["cache_hits"], 1)
        self.assertEqual(phases["idle_control"]["turns"], 1)
        self.assertEqual(phases["unknown"]["phase_reasons"][0]["value"], "insufficient-metadata")
        self.assertGreater(phases["planning"]["input_tokens_est"], 0)
        self.assertGreaterEqual(phases["planning"]["cost_est_usd"], 0)
        sample_phases = {row["workflow_phase"] for row in result["recent_samples"]}
        self.assertIn("planning", sample_phases)
        self.assertFalse(result["privacy"]["raw_params_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/codex-effectiveness?limit=20")
        self.assertEqual(response.status_code, 200)
        endpoint_payload = response.json()
        endpoint_phases = {row["phase"] for row in endpoint_payload["workflow_phase_breakdown"]}
        self.assertEqual(endpoint_phases, set(phases))

    def test_codex_effectiveness_reports_repeated_context_plateau_candidates(self):
        raw_prompt_text = "raw prompt must not appear in plateau report"
        sizes = [10_000, 10_100, 9_950, 10_150]
        saved = [10, 20, 0, 0]
        for index, input_text_chars in enumerate(sizes):
            request_id = f"req-plateau-{index}"
            server.store.log_codex_app_event(
                id=f"start-plateau-{index}",
                created_at=f"2026-06-08T12:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id="thread-plateau-candidate",
                message_chars=input_text_chars + 100,
                params_chars=input_text_chars + 50,
                input_items=2,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-plateau-candidate",
                routing_json=stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                    "policy_source": "local-default",
                }),
                crunch_json=stable_json({
                    "status": "applied" if saved[index] else "skipped",
                    "reason": "codex-repeated-scaffolding-crunched" if saved[index] else "no-change",
                    "applied": bool(saved[index]),
                    "changed": bool(saved[index]),
                    "saved_chars": saved[index],
                    "tokens_saved_est": saved[index] // 4,
                }),
                cache_json=stable_json({
                    "status": "miss",
                    "reason": "exact-miss",
                    "eligible": True,
                    "policy_source": "local-default",
                }),
                event_window_json=stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 3,
                    "method_counts": {"turn/start": 1, "item/agentMessage/delta": 2},
                    "direction_counts": {"client_to_server": 1, "server_to_client": 2},
                    "input_text_chars": input_text_chars,
                    "debug_prompt": raw_prompt_text,
                }),
            )
            server.store.log_codex_app_event(
                id=f"end-plateau-{index}",
                created_at=f"2026-06-08T12:00:1{index}+00:00",
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id="thread-plateau-candidate",
                message_chars=180,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=160,
                error_code=None,
                error_message=None,
                latency_ms=100 + index,
                session_id="session-plateau-candidate",
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        report = result["repeated_context_plateau_candidates"]
        [candidate] = report["candidates"]
        encoded = json.dumps(result)

        self.assertEqual(result["summary"]["repeated_context_plateau_candidate_count"], 1)
        self.assertEqual(candidate["scope_id"], "thread-plateau-candidate")
        self.assertEqual(candidate["scope_basis"], "thread_id")
        self.assertEqual(candidate["turns"], 4)
        self.assertEqual(candidate["plateau_count"], 3)
        self.assertEqual(candidate["candidate_pairs"], 3)
        self.assertEqual(candidate["median_input_chars"], 10_050)
        self.assertEqual(candidate["p90_input_chars"], 10_150)
        self.assertEqual(candidate["current_saved_chars"], 30)
        self.assertGreater(candidate["estimated_opportunity_saved_chars"], 0)
        self.assertGreater(candidate["estimated_opportunity_tokens"], 0)
        self.assertEqual(report["policy"]["min_input_chars"], 8_000)
        self.assertFalse(result["privacy"]["raw_params_included"])
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("debug_prompt", encoded)

    def test_codex_effectiveness_uses_persisted_metadata_only_event_window(self):
        secret = "raw prompt must not appear"
        server.store.log_codex_app_event(
            id="start-window",
            created_at="2026-06-08T10:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="req-window",
            thread_id="thread-window",
            message_chars=240,
            params_chars=180,
            input_items=1,
            input_text_chars=96,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="session-window",
            routing_json=stable_json({
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
            }),
            crunch_json=stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
            cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False}),
            event_window_json=stable_json({
                "schema": "agentflow.codex_app_event_window.v1",
                "start_event_id": "start-window",
                "created_at": "2026-06-08T10:00:00+00:00",
                "session_id": "session-window",
                "request_id": "req-window",
                "thread_id": "thread-window",
                "event_count": 4,
                "method_counts": {
                    "turn/start": 1,
                    "turn/diff/updated": 2,
                    "turn/completed": 1,
                },
                "direction_counts": {"client_to_server": 1, "server_to_client": 3},
                "first_event_delta_ms": 0,
                "last_event_delta_ms": 1200,
                "input_items": 1,
                "input_text_chars": 96,
                "start_message_chars": 240,
                "start_params_chars": 180,
                "result_chars": 52,
                "server_message_chars": 300,
                "error_count": 0,
                "model_field_state": "absent",
                "debug_prompt": secret,
            }),
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=5))

        self.assertEqual(result["summary"]["workflow_phase_known"], 1)
        self.assertEqual(result["workflow_phase_breakdown"][0]["phase"], "verification")
        self.assertEqual(result["workflow_phase_source_breakdown"][0]["value"], "event_window")
        sample = result["recent_samples"][0]
        self.assertEqual(sample["workflow_phase_source"], "event_window")
        self.assertEqual(sample["workflow_phase_reason"], "event-window-signal:verification")
        self.assertEqual(sample["event_window"]["event_count"], 4)
        self.assertEqual(sample["event_window"]["model_field_state"], "absent")
        self.assertTrue(sample["event_window"]["request_id_present"])
        self.assertNotIn("debug_prompt", json.dumps(sample["event_window"]))
        self.assertNotIn(secret, json.dumps(result))
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_codex_effectiveness_uses_event_window_when_decision_phase_unknown(self):
        server.store.log_codex_app_event(
            id="start-window-phase-fallback",
            created_at="2026-06-08T10:01:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="req-window-phase-fallback",
            thread_id="thread-window-phase-fallback",
            message_chars=240,
            params_chars=180,
            input_items=1,
            input_text_chars=96,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="session-window-phase-fallback",
            routing_json=stable_json({
                "status": "skipped",
                "reason": "summary-model-hint-not-applied",
                "workflow_phase": "unknown",
                "workflow_phase_reason": "workflow-phase-not-summary",
                "applied": False,
                "policy_source": "local-default",
            }),
            crunch_json=stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
            cache_json=stable_json({
                "status": "skipped",
                "reason": "codex-app-cache-disabled",
                "eligible": False,
                "workflow_phase": "unknown",
            }),
            event_window_json=stable_json({
                "schema": "agentflow.codex_app_event_window.v1",
                "event_count": 3,
                "method_counts": {
                    "turn/start": 1,
                    "item/commandExecution/outputDelta": 1,
                    "turn/completed": 1,
                },
                "direction_counts": {"client_to_server": 1, "server_to_client": 2},
                "workflow_phase": "tool_execution",
                "workflow_phase_reason": "event-window-signal:tool_execution",
                "workflow_phase_source": "event_window",
                "workflow_phase_confidence": "high",
                "workflow_phase_signals": ["item/commandExecution/outputDelta"],
                "model_field_state": "present",
            }),
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=5))

        self.assertEqual(result["summary"]["workflow_phase_known"], 1)
        self.assertEqual(result["workflow_phase_breakdown"][0]["phase"], "tool_execution")
        self.assertEqual(result["workflow_phase_source_breakdown"][0]["value"], "event_window")
        sample = result["recent_samples"][0]
        self.assertEqual(sample["workflow_phase"], "tool_execution")
        self.assertEqual(sample["workflow_phase_source"], "event_window")
        self.assertEqual(sample["workflow_phase_signals"], ["item/commandExecution/outputDelta"])
        self.assertEqual(sample["event_window"]["workflow_phase"], "tool_execution")
        self.assertEqual(sample["event_window"]["workflow_phase_confidence"], "high")

    def test_old_context_summary_stats_are_attributed_separately(self):
        for cache_hit, cost in ((False, 0.0002), (True, 0.0)):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=1_000,
                output_tokens_est=0,
                actual_input_tokens=1_000,
                actual_output_tokens=0,
                cost_est_usd=cost,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({
                    "changed": False,
                    "old_context_summarization": {
                        "status": "applied",
                        "reason": "summary-cache-hit" if cache_hit else "summary-created",
                        "summary_cache_hit": cache_hit,
                        "summary_cost_est_usd": cost,
                        "saved_chars": 2_000,
                        "tokens_saved_est": 500,
                    },
                }),
                routing_json=None,
                cache_json=None,
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-summary",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]

        self.assertEqual(summary["old_context_summary_applied_count"], 2)
        self.assertEqual(summary["old_context_summary_created_count"], 1)
        self.assertEqual(summary["old_context_summary_cache_hits"], 1)
        self.assertAlmostEqual(summary["old_context_summary_cache_hit_rate"], 0.5, places=6)
        self.assertEqual(summary["old_context_summary_tokens_saved"], 1_000)
        self.assertAlmostEqual(summary["old_context_summary_cost_usd"], 0.0002, places=6)
        self.assertAlmostEqual(summary["old_context_summary_savings_usd"], 0.003, places=6)
        self.assertAlmostEqual(summary["today_old_context_summary_net_usd"], 0.0028, places=6)

    def test_old_context_summary_opportunity_counts_eligible_skips_and_net_projection(self):
        rows = [
            {
                "status": "planned",
                "reason": "eligible",
                "eligible_turns": 4,
                "eligible_chars": 24_000,
            },
            {
                "status": "applied",
                "reason": "summary-created",
                "summary_cache_hit": False,
                "summary_input_tokens": 1_000,
                "summary_output_tokens": 120,
                "summary_cost_est_usd": 0.0004,
                "tokens_saved_est": 2_000,
            },
            {
                "status": "skipped",
                "reason": "disabled",
            },
            {
                "status": "skipped",
                "reason": "eligible-context-too-small",
                "eligible_turns": 0,
                "eligible_chars": 0,
            },
        ]
        for meta in rows:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=8_000,
                output_tokens_est=0,
                actual_input_tokens=8_000,
                actual_output_tokens=0,
                cost_est_usd=0.0,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({
                    "changed": False,
                    "old_context_summarization": meta,
                }),
                routing_json=None,
                cache_json=None,
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-summary-opportunity",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_old_context_summary(server.store))
        full = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]

        self.assertEqual(summary["observed_rows"], 4)
        self.assertEqual(summary["eligible_rows"], 2)
        self.assertEqual(summary["ineligible_rows"], 2)
        self.assertEqual(summary["planned_rows"], 1)
        self.assertEqual(summary["applied_rows"], 1)
        self.assertEqual(summary["summary_created_rows"], 1)
        self.assertGreater(summary["gross_saved_tokens_est"], 0)
        self.assertGreater(summary["gross_savings_usd"], 0)
        self.assertGreater(summary["summary_model_cost_usd"], 0)
        self.assertGreater(summary["net_savings_usd"], 0)
        self.assertGreater(summary["payback_ratio"], 1)
        self.assertIn("old_context_summary_opportunity", full)
        self.assertFalse(result["privacy"]["raw_old_context_included"])
        self.assertFalse(result["privacy"]["generated_summaries_included"])
        reasons = {row["value"]: row["count"] for row in result["skip_reason_breakdown"]}
        self.assertEqual(reasons["disabled"], 1)
        self.assertEqual(reasons["tool/protocol-context-only"], 1)

    def test_old_context_summary_endpoint_and_dashboard_panel_are_read_only_metadata(self):
        secret = "raw old context must stay out"
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=8_000,
            output_tokens_est=0,
            actual_input_tokens=8_000,
            actual_output_tokens=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({
                "changed": False,
                "old_context_summarization": {
                    "status": "skipped",
                    "reason": "disabled",
                    "debug_context": secret,
                },
            }),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-summary-route",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=server.store.path,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
        )
        with TestClient(app) as client:
            payload = client.get("/agentflow/stats/old-context-summary").json()
            html = client.get("/agentflow/dashboard").text

        self.assertEqual(payload["summary"]["skipped_rows"], 1)
        self.assertIn("Old-context summarization opportunity", html)
        self.assertIn("old-context-summary-tbody", html)
        self.assertIn("raw context omitted", html)
        rendered = json.dumps(payload) + html
        self.assertNotIn(secret, rendered)

    def test_old_context_summary_readiness_and_plateau_category_impact_are_served_metadata(self):
        secret = "raw old-context readiness context must stay out"

        def log_summary_row(
            suffix: str,
            meta: dict[str, object],
            *,
            session_id: str,
            category: str,
            status_code: int = 200,
            retry_count: int = 0,
            text_chars: int = 32_000,
        ) -> None:
            server.store.log_call(
                id=f"old-context-readiness-{suffix}",
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=status_code,
                latency_ms=1000,
                input_tokens_est=8_000,
                output_tokens_est=100,
                actual_input_tokens=8_000,
                actual_output_tokens=100,
                cost_est_usd=0.02,
                cost_baseline_usd=0.03,
                crunch_json=stable_json({
                    "changed": meta.get("status") == "applied",
                    "old_context_summarization": {
                        **meta,
                        "debug_context": secret,
                    },
                }),
                routing_json=stable_json({"category": category, "text_chars": text_chars}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error="raw old-context readiness error must stay out" if status_code >= 400 else None,
                request_json=stable_json({"messages": [{"content": "raw old-context readiness request"}]}),
                response_json=stable_json({"content": "generated old-context readiness summary"}),
                session_id=session_id,
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
            )

        base = {
            "enabled": True,
            "rule_id": "old-context-dashboard-rule",
            "candidate_id": "old-context-dashboard-candidate",
            "policy_source": "managed-recommended",
            "model": "claude-haiku-4-5-20251001",
            "category": "tool-result",
            "eligible_chars": 36_000,
            "eligible_turns": 4,
            "canary": {
                "enabled": True,
                "fraction": 0.5,
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 2,
                "min_canary_applied_samples": 1,
                "min_canary_holdout_samples": 1,
                "rollback_error_rate": 0.4,
            },
        }
        log_summary_row(
            "disabled",
            {"enabled": False, "status": "skipped", "reason": "disabled", "category": "chat"},
            session_id="old-context-readiness-session-secret-a",
            category="chat",
            text_chars=12_000,
        )
        log_summary_row(
            "eligible",
            {**base, "status": "planned", "reason": "eligible", "eligible_chars": 24_000},
            session_id="old-context-readiness-session-secret-a",
            category="tool-result",
            text_chars=24_000,
        )
        log_summary_row(
            "applied-rollback",
            {
                **base,
                "status": "applied",
                "reason": "summary-created",
                "tokens_saved_est": 1_500,
                "summary_cost_est_usd": 0.001,
                "estimated_net_savings_usd": 0.0035,
                "summary_status_code": 500,
                "canary": {**base["canary"], "cohort": "canary_applied", "selected": True},
            },
            session_id="old-context-readiness-session-secret-b",
            category="tool-result",
            status_code=500,
            retry_count=1,
            text_chars=40_000,
        )
        log_summary_row(
            "holdout",
            {
                **base,
                "status": "skipped",
                "reason": "canary_holdout",
                "eligible_chars": 34_000,
                "canary": {**base["canary"], "cohort": "canary_holdout", "selected": False},
            },
            session_id="old-context-readiness-session-secret-b",
            category="tool-result",
            text_chars=34_000,
        )
        log_summary_row(
            "safety-stop",
            {
                **base,
                "status": "skipped",
                "reason": "safety-stop",
                "safety_stop_state": "stopped",
            },
            session_id="old-context-readiness-session-secret-b",
            category="tool-result",
            text_chars=36_000,
        )

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=server.store.path,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
        )
        with TestClient(app) as client:
            payload = client.get("/agentflow/stats/old-context-summary").json()
            html = client.get("/agentflow/dashboard").text

        readiness_counts = {row["value"]: row["count"] for row in payload["readiness"]["state_breakdown"]}
        self.assertEqual(readiness_counts["disabled"], 1)
        self.assertEqual(readiness_counts["eligible"], 2)
        self.assertEqual(readiness_counts["applied"], 1)
        self.assertEqual(readiness_counts["holdout"], 1)
        self.assertEqual(readiness_counts["safety_stop"], 1)
        self.assertEqual(readiness_counts["rollback"], 1)
        self.assertEqual(payload["readiness"]["latest_quality_gate_verdict"], "rollback")

        plateau = payload["plateau_session_context"]
        self.assertEqual(plateau["affected_session_count"], 2)
        self.assertGreater(plateau["median_text_chars"], 0)
        self.assertGreater(plateau["p90_text_chars"], 0)
        by_category = {row["category"]: row for row in plateau["category_breakdown"]}
        self.assertEqual(by_category["tool-result"]["observed_rows"], 4)
        self.assertEqual(by_category["tool-result"]["holdout_rows"], 1)
        self.assertEqual(by_category["tool-result"]["safety_stop_rows"], 1)
        self.assertGreater(by_category["tool-result"]["projected_saved_tokens_est"], 0)
        self.assertGreater(by_category["tool-result"]["applied_saved_tokens_est"], 0)

        self.assertIn("Old-context summary readiness states", html)
        self.assertIn("old-context-summary-readiness-tbody", html)
        self.assertIn("Old-context plateau impact by category", html)
        self.assertIn("old-context-summary-plateau-categories-tbody", html)
        rendered = json.dumps(payload, sort_keys=True) + html
        self.assertNotIn(secret, rendered)
        self.assertNotIn("old-context-readiness-session-secret", rendered)
        self.assertNotIn("raw old-context readiness request", rendered)
        self.assertNotIn("generated old-context readiness summary", rendered)
        self.assertNotIn("raw old-context readiness error", rendered)

    def _log_old_context_summary_quality_row(
        self,
        *,
        candidate_id: str,
        suffix: str,
        cohort: str,
        status_code: int = 200,
        retry_count: int = 0,
        latency_ms: int = 1000,
        secret: str = "raw quality gate secret must stay out",
    ):
        applied = cohort == "canary_applied"
        meta = {
            "enabled": True,
            "status": "applied" if applied else "skipped",
            "reason": "summary-created" if applied else "canary_holdout",
            "rule_id": f"rule-{candidate_id}",
            "candidate_id": candidate_id,
            "policy_source": "managed-recommended",
            "model": "claude-haiku-4-5-20251001",
            "category": "chat",
            "eligible_chars": 32_000,
            "eligible_turns": 3,
            "debug_context": secret,
            "canary": {
                "enabled": True,
                "cohort": cohort,
                "selected": applied,
                "fraction": 0.5,
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 4,
                "min_canary_applied_samples": 2,
                "min_canary_holdout_samples": 2,
                "max_error_rate": 0.05,
                "max_error_rate_delta": 0.0,
                "max_retry_rate": 0.25,
                "max_summary_failure_rate": 0.02,
                "rollback_error_rate": 0.4,
                "rollback_summary_failure_rate": 0.2,
            },
        }
        if applied:
            meta.update({
                "before_chars": 40_000,
                "saved_chars": 4_000,
                "tokens_saved_est": 1_000,
                "estimated_gross_savings_usd": 0.003,
                "summary_cost_est_usd": 0.001,
                "estimated_net_savings_usd": 0.002,
                "summary_status_code": 200 if status_code < 400 else status_code,
                "summary_cache_hit": False,
            })
        if status_code >= 400:
            meta["summary_error"] = "summary failure bucket"
        server.store.log_call(
            id=f"dashboard-quality-{candidate_id}-{suffix}",
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=10_000,
            output_tokens_est=200,
            actual_input_tokens=9_000,
            actual_output_tokens=180,
            cost_est_usd=0.03,
            cost_baseline_usd=0.04,
            crunch_json=stable_json({
                "changed": applied,
                "old_context_summarization": meta,
            }),
            routing_json=stable_json({"category": "chat", "text_chars": 40_000}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error="raw quality error must not leak" if status_code >= 400 else None,
            request_json=stable_json({"messages": [{"content": "raw quality request must not leak"}]}),
            response_json=stable_json({"content": "generated quality summary must not leak"}),
            session_id="quality-gate-session-secret",
            category="chat",
            retry_count=retry_count,
            provider="anthropic",
        )

    def test_old_context_summary_quality_gate_dashboard_reports_verdicts_and_cohort_deltas(self):
        for suffix, cohort, status_code, retry_count, latency in (
            ("a0", "canary_applied", 200, 0, 1000),
            ("a1", "canary_applied", 200, 0, 1100),
            ("h0", "canary_holdout", 200, 0, 1000),
            ("h1", "canary_holdout", 200, 0, 1100),
        ):
            self._log_old_context_summary_quality_row(
                candidate_id="candidate-promote",
                suffix=suffix,
                cohort=cohort,
                status_code=status_code,
                retry_count=retry_count,
                latency_ms=latency,
            )
        for suffix, cohort, retry_count in (
            ("a0", "canary_applied", 1),
            ("a1", "canary_applied", 1),
            ("h0", "canary_holdout", 0),
            ("h1", "canary_holdout", 0),
        ):
            self._log_old_context_summary_quality_row(
                candidate_id="candidate-hold",
                suffix=suffix,
                cohort=cohort,
                retry_count=retry_count,
            )
        for suffix, cohort, status_code in (
            ("a0", "canary_applied", 500),
            ("a1", "canary_applied", 200),
            ("h0", "canary_holdout", 200),
            ("h1", "canary_holdout", 200),
        ):
            self._log_old_context_summary_quality_row(
                candidate_id="candidate-rollback",
                suffix=suffix,
                cohort=cohort,
                status_code=status_code,
            )
        self._log_old_context_summary_quality_row(
            candidate_id="candidate-insufficient",
            suffix="a0",
            cohort="canary_applied",
        )

        payload = asyncio.run(stats_views.stats_old_context_summary(server.store))
        by_candidate = {row["candidate_id"]: row for row in payload["quality_gates"]}

        self.assertEqual(by_candidate["candidate-promote"]["verdict"], "promote")
        self.assertIn("quality-gate-passed", by_candidate["candidate-promote"]["reason_codes"])
        self.assertEqual(by_candidate["candidate-promote"]["metrics"]["canary_applied_count"], 2)
        self.assertEqual(by_candidate["candidate-promote"]["metrics"]["canary_holdout_count"], 2)
        self.assertEqual(by_candidate["candidate-promote"]["metrics"]["applied_minus_holdout_error_rate"], 0.0)
        self.assertEqual(by_candidate["candidate-promote"]["metrics"]["applied_minus_holdout_retry_rate"], 0.0)
        self.assertEqual(by_candidate["candidate-promote"]["metrics"]["applied_minus_holdout_latency_avg_ms"], 0.0)
        self.assertGreater(by_candidate["candidate-promote"]["metrics"]["actual_net_savings_usd"], 0)
        self.assertGreater(by_candidate["candidate-promote"]["metrics"]["payback_ratio"], 1)

        self.assertEqual(by_candidate["candidate-hold"]["verdict"], "hold")
        self.assertIn("applied-retry-rate-above-threshold", by_candidate["candidate-hold"]["reason_codes"])
        self.assertGreater(by_candidate["candidate-hold"]["metrics"]["applied_minus_holdout_retry_rate"], 0)

        self.assertEqual(by_candidate["candidate-rollback"]["verdict"], "rollback")
        self.assertIn("rollback-error-rate", by_candidate["candidate-rollback"]["reason_codes"])
        self.assertEqual(by_candidate["candidate-rollback"]["metrics"]["summary_failure_count"], 1)

        self.assertEqual(by_candidate["candidate-insufficient"]["verdict"], "insufficient-evidence")
        self.assertIn("insufficient-matched-samples", by_candidate["candidate-insufficient"]["reason_codes"])

        summary = payload["quality_gate_summary"]
        self.assertEqual(summary["promote_count"], 1)
        self.assertEqual(summary["hold_count"], 1)
        self.assertEqual(summary["rollback_count"], 1)
        self.assertEqual(summary["insufficient_evidence_count"], 1)
        self.assertIn("reason_code_breakdown", summary)
        self.assertFalse(payload["privacy"]["raw_old_context_included"])
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["local_session_ids_included"])

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw quality request", rendered)
        self.assertNotIn("generated quality summary", rendered)
        self.assertNotIn("quality-gate-session-secret", rendered)
        self.assertNotIn("raw quality error", rendered)
        self.assertNotIn("raw quality gate secret", rendered)

    def test_old_context_summary_quality_gate_dashboard_html_is_read_only_metadata(self):
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=server.store.path,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
        )
        with TestClient(app) as client:
            html = client.get("/agentflow/dashboard").text

        self.assertIn("Old-context summary quality gates", html)
        self.assertIn("old-context-summary-quality-tbody", html)
        self.assertIn("Blocking reasons", html)
        self.assertNotIn("textarea", html.lower())
        self.assertNotIn("form method", html.lower())

    def test_old_context_summary_rollout_health_reports_canary_safety_and_queue(self):
        for suffix, cohort in (
            ("a0", "canary_applied"),
            ("a1", "canary_applied"),
            ("h0", "canary_holdout"),
            ("h1", "canary_holdout"),
        ):
            self._log_old_context_summary_quality_row(
                candidate_id="candidate-rollout-health",
                suffix=suffix,
                cohort=cohort,
            )
        server.store.enqueue_managed_outcome_feedback(
            id="summary-feedback-health",
            source_surface="old_context_summary_outcome",
            endpoint="policy-events",
            optimization_unit_id=123,
            payload_json=stable_json({"raw_payload": "raw rollout queue secret must not leak"}),
            status="queued",
        )

        payload = asyncio.run(stats_views.stats_old_context_summary(server.store))
        health = payload["rollout_health"]

        self.assertEqual(health["schema"], "agentflow.old_context_summary_rollout_health.v1")
        self.assertEqual(health["status"], "canary-observed")
        self.assertEqual(health["latest"]["candidate_id"], "candidate-rollout-health")
        self.assertEqual(health["rollout_counts"]["canary_applied_rows"], 2)
        self.assertEqual(health["rollout_counts"]["canary_holdout_rows"], 2)
        self.assertEqual(health["rollout_counts"]["safety_stop_rows"], 0)
        self.assertGreater(health["economics"]["net_savings_usd"], 0)
        self.assertGreater(health["economics"]["payback_ratio"], 1)
        self.assertEqual(health["managed_feedback_queue"]["summary"]["queued"], 1)
        self.assertEqual(health["managed_feedback_queue"]["summary"]["due"], 1)
        self.assertFalse(health["managed_feedback_queue"]["privacy"]["payload_json_included"])
        self.assertFalse(health["privacy"]["raw_old_context_included"])
        self.assertFalse(health["privacy"]["local_session_ids_included"])

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw rollout queue secret", rendered)
        self.assertNotIn("quality-gate-session-secret", rendered)

    def test_old_context_summary_rollout_health_dashboard_tab_is_metadata_only(self):
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=server.store.path,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
        )
        with TestClient(app) as client:
            dashboard = client.get("/agentflow/dashboard")

        html = dashboard.text
        self.assertIn("Old-context summary", html)
        self.assertIn("Old-context summarization rollout health", html)
        self.assertIn("old-context-summary-rollout-tbody", html)
        self.assertIn("old-context-summary-feedback-tbody", html)
        self.assertIn("rollout_health", html)
        self.assertIn("raw context omitted", html)
        self.assertIn("payload omitted", html)
        self.assertNotIn("textarea", html.lower())
        self.assertNotIn("form method", html.lower())

    def test_phase_routing_dashboard_reports_canary_safety_savings_and_feedback(self):
        secret = "raw phase routing payload must not leak"

        async def seeded_policy_state():
            return {
                "schema": "agentflow.policy_state.v1",
                "routing": {
                    "enabled": True,
                    "policy_source": "managed-recommended",
                    "rule_path": "/tmp/phase-routing-rules.yaml",
                    "file": {"reload_required": False},
                    "phase_canary": {
                        "enabled": True,
                        "policy_id": "candidate-phase-route",
                        "model_pattern": "sonnet",
                        "target_model": "haiku",
                        "eligible_workflow_phases": ["tool-execution", "summary"],
                        "excluded_workflow_phases": ["planning", "thinking", "unknown"],
                        "min_workflow_phase_confidence": "medium",
                        "canary_fraction": 0.5,
                        "holdout_fraction": 0.25,
                        "safety_stop": {
                            "enabled": True,
                            "window_hours": 24,
                            "min_samples": 2,
                            "min_holdout_samples": 1,
                            "max_error_rate": 0.05,
                            "max_retry_rate": 0.2,
                            "max_fallback_rate": 0.2,
                        },
                    },
                },
                "summary": {},
            }

        def log_phase_call(
            suffix,
            *,
            routed_model,
            status,
            cohort,
            cost,
            baseline,
            status_code=200,
            retry_count=0,
            latency_ms=1000,
            safety=None,
        ):
            server.store.log_call(
                id=f"phase-routing-{suffix}",
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model=routed_model,
                stream=1,
                cache_hit=0,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens_est=2_000,
                output_tokens_est=200,
                actual_input_tokens=2_000,
                actual_output_tokens=200,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
                routing_json=stable_json({
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "workflow_phase_confidence": "high",
                    "text_chars": 8000,
                    "has_tools": True,
                    "reason": "phase canary selected Sonnet-to-Haiku route" if status == "applied" else "phase canary holdout; keep requested model",
                    "phase_canary": {
                        "enabled": True,
                        "policy_id": "candidate-phase-route",
                        "status": status,
                        "cohort": cohort,
                        "reason": "safety-stop-tripped" if status == "safety_stopped" else ("selected-canary" if status == "applied" else "selected-holdout"),
                        "target_model": "claude-haiku-4-5-20251001",
                        "workflow_phase": "tool-execution",
                        "workflow_phase_confidence": "high",
                        "category": "tool-result",
                        "text_bucket": "2k-8k",
                        "has_tools": True,
                        "canary_fraction": 0.5,
                        "holdout_fraction": 0.25,
                        "policy_source": "managed-recommended",
                        "debug_context": secret,
                        "safety_stop": safety or {"enabled": True, "status": "evaluated", "tripped": False, "reason_codes": [], "sample_count": 2, "holdout_sample_count": 1},
                    },
                    "phase_routing_feedback": {
                        "enabled": True,
                        "status": "queued",
                        "reason": "queued",
                        "source_surface": "phase_routing_outcome",
                        "queue_id": "phase-queue-secret",
                        "payload_included": False,
                    },
                }),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=secret if status_code >= 400 else None,
                request_json=stable_json({"messages": [{"content": secret}]}),
                response_json=stable_json({"content": secret}),
                session_id="phase-routing-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        log_phase_call("applied", routed_model="claude-haiku-4-5-20251001", status="applied", cohort="applied", cost=0.005, baseline=0.02)
        log_phase_call("holdout", routed_model="claude-sonnet-4-6", status="holdout", cohort="holdout", cost=0.02, baseline=0.02, latency_ms=1200)
        log_phase_call(
            "stopped",
            routed_model="claude-sonnet-4-6",
            status="safety_stopped",
            cohort="stopped",
            cost=0.02,
            baseline=0.02,
            status_code=429,
            retry_count=1,
            safety={"enabled": True, "status": "tripped", "tripped": True, "reason_codes": ["error-rate"], "sample_count": 3, "holdout_sample_count": 1},
        )
        server.store.enqueue_managed_outcome_feedback(
            id="phase-outcome-feedback",
            source_surface="phase_routing_outcome",
            endpoint="/v1/policy-events",
            optimization_unit_id=123,
            payload_json=stable_json({"raw_payload": secret}),
            status="queued",
        )
        server.store.enqueue_managed_outcome_feedback(
            id="phase-lifecycle-feedback",
            source_surface="phase_routing_lifecycle",
            endpoint="/v1/policy-events",
            optimization_unit_id=0,
            payload_json=stable_json({
                "event_type": "dry-run",
                "recommendation_id": "phase-routing:secret-id",
                "metadata": {
                    "schema": "agentflow.phase_routing_lifecycle_metadata.v1",
                    "command": "phase-routing-dry-run",
                    "local_result_status": "ok",
                    "dry_run": True,
                    "read_only": True,
                    "policy_source": "managed-recommended",
                    "sampled_call_count": 3,
                    "matched_count": 2,
                    "projected_candidate_count": 1,
                    "excluded_count": 1,
                    "projected_savings_usd": 0.015,
                    "risk_warning_count": 1,
                    "candidate_rule_ids": ["candidate-phase-route"],
                    "excluded_count_by_reason": {"thinking": 1},
                    "raw_payload": secret,
                },
            }),
            status="queued",
        )

        with patch.object(stats_views, "stats_policies", seeded_policy_state):
            payload = asyncio.run(stats_views.stats_phase_routing(server.store, limit=50))
            app = create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.anthropic.com",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            with TestClient(app) as client:
                endpoint = client.get("/agentflow/stats/phase-routing?limit=50")
                html = client.get("/agentflow/dashboard").text

        self.assertEqual(payload["schema"], "agentflow.phase_routing_dashboard.v1")
        self.assertEqual(payload["status"], "safety-stopped")
        self.assertEqual(payload["summary"]["canary_applied_rows"], 1)
        self.assertEqual(payload["summary"]["canary_holdout_rows"], 1)
        self.assertEqual(payload["summary"]["safety_stop_rows"], 1)
        self.assertGreater(payload["summary"]["projected_savings_usd"], 0)
        self.assertGreater(payload["summary"]["observed_savings_usd"], 0)
        self.assertEqual(payload["managed_feedback_queue"]["summary"]["queued"], 1)
        self.assertEqual(payload["lifecycle"]["summary"]["feedback_count"], 1)
        self.assertEqual(payload["lifecycle"]["summary"]["latest_dry_run_matched_count"], 2)
        self.assertTrue(payload["safety_stop"]["active"])
        self.assertFalse(payload["privacy"]["local_session_ids_included"])
        self.assertFalse(payload["privacy"]["queue_payload_json_included"])
        self.assertEqual(endpoint.status_code, 200)
        self.assertEqual(endpoint.json()["schema"], "agentflow.phase_routing_dashboard.v1")
        self.assertIn("Phase-routing rollout health", html)
        self.assertIn("phase-routing-opportunity-tbody", html)
        self.assertIn("phase-routing-canary-tbody", html)
        self.assertIn("raw prompts omitted", html)

        rendered = json.dumps(payload, sort_keys=True) + html
        self.assertNotIn(secret, rendered)
        self.assertNotIn("phase-routing-session-secret", rendered)
        self.assertNotIn("phase-queue-secret", rendered)

    def test_cache_decision_breakdown_groups_status_reason_and_hit_type(self):
        rows = [
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            {"status": "miss", "reason": "file-dependency-changed", "policy_source": "local-default"},
            {"status": "hit", "reason": "exact-match", "hit_type": "exact", "policy_source": "local-default"},
        ]
        for cache_json in rows:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1 if cache_json["reason"] == "streaming" else 0,
                cache_hit=1 if cache_json["status"] == "hit" else 0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=0.0,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({"changed": False}),
                routing_json=None,
                cache_json=stable_json(cache_json),
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-cache",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        breakdown = {
            (row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in result["cache_decision_breakdown"]
        }

        self.assertEqual(breakdown[("skipped", "streaming", "")], 2)
        self.assertEqual(breakdown[("miss", "exact-miss", "")], 1)
        self.assertEqual(breakdown[("miss", "file-dependency-changed", "")], 1)
        self.assertEqual(breakdown[("hit", "exact-match", "exact")], 1)
        json.dumps(result["cache_decision_breakdown"])

    def test_cache_zero_hit_blocker_ladder_ranks_provider_surface_blockers_without_raw_data(self):
        secret_prompt = "raw cache blocker prompt must not leak"
        secret_session = "cache-blocker-session-secret"
        secret_cache_key = "cache-key-secret"
        secret_path = "/tmp/cache-blocker-secret.py"

        def log_cache_row(index, *, provider, path, source_surface, endpoint, stream, has_tools, cache_json, routing_json=None):
            routing = {
                "category": "chat",
                "has_tools": has_tools,
                "openai_feature_unit": {
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "replayability_level": cache_json.get("replayability_level", "features_only"),
                },
            }
            if routing_json:
                routing.update(routing_json)
            server.store.log_call(
                id=f"cache-blocker-ladder-{index}",
                created_at=utc_now(),
                path=path,
                requested_model="gpt-5.4-mini" if provider == "openai" else "claude-sonnet-4-6",
                routed_model="gpt-5.4-mini" if provider == "openai" else "claude-sonnet-4-6",
                stream=1 if stream else 0,
                cache_hit=1 if cache_json.get("status") == "hit" else 0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=0.0,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing),
                cache_json=stable_json({
                    **cache_json,
                    "cache_key": secret_cache_key,
                    "file_path": secret_path,
                }),
                error=None,
                request_json=stable_json({"input": secret_prompt}),
                response_json=stable_json({"output": "raw cache blocker response must not leak"}),
                session_id=secret_session,
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider=provider,
                source_surface=source_surface,
                endpoint=endpoint,
                requested_model_family="gpt-5" if provider == "openai" else "claude",
                routed_model_family="gpt-5" if provider == "openai" else "claude",
            )

        fixtures = [
            ("openai", "/v1/responses", "openai_responses", "responses", True, False, {"status": "skipped", "reason": "streaming", "policy_source": "local-default", "replayability_level": "features_only"}),
            ("openai", "/v1/responses", "openai_responses", "responses", True, False, {"status": "skipped", "reason": "streaming", "policy_source": "local-default", "replayability_level": "features_only"}),
            ("anthropic", "/v1/messages", "anthropic_messages", "messages", False, True, {"status": "skipped", "reason": "tools-disabled", "policy_source": "local-default", "replayability_level": "local-exact-response"}),
            ("openai", "/v1/chat/completions", "openai_chat", "chat", False, False, {"status": "skipped", "reason": "cache-disabled", "policy_source": "local-default"}),
            ("openai", "/v1/responses", "openai_responses", "responses", False, False, {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            ("openai", "/v1/responses", "openai_responses", "responses", False, False, {"status": "holdout", "reason": "canary_holdout", "policy_source": "local-manual"}),
            ("openai", "/v1/responses", "openai_responses", "responses", False, True, {"status": "miss", "reason": "dependency-changed", "policy_source": "local-manual", "file_dependency_audit": {"invalidation_reason": "dependency-changed", "safe_invalidation_evidence": False}}),
            ("openai", "/v1/responses", "openai_responses", "responses", False, False, {"status": "skipped", "reason": "exact-miss", "policy_source": "local-manual", "policy_reload_required": True}),
        ]
        for index, fixture in enumerate(fixtures):
            log_cache_row(index, provider=fixture[0], path=fixture[1], source_surface=fixture[2], endpoint=fixture[3], stream=fixture[4], has_tools=fixture[5], cache_json=fixture[6])

        result = asyncio.run(stats_views.stats_full(server.store))
        ladder_payload = result["cache_zero_hit_blocker_ladder"]
        ladder = ladder_payload["ladder"]
        by_code = {row["blocker_code"]: row for row in ladder}

        self.assertEqual(ladder_payload["schema"], "agentflow.cache_zero_hit_blocker_ladder.v1")
        self.assertTrue(ladder_payload["summary"]["zero_hit_window"])
        self.assertEqual(ladder[0]["blocker_code"], "skipped-streaming")
        self.assertEqual(ladder[0]["provider"], "openai")
        self.assertEqual(ladder[0]["source_surface"], "openai_responses")
        self.assertEqual(ladder[0]["endpoint"], "responses")
        self.assertEqual(ladder[0]["stream_mode"], "stream")
        self.assertEqual(ladder[0]["tool_presence"], "no-tools")
        self.assertEqual(ladder[0]["replayability_level"], "features_only")
        self.assertEqual(ladder[0]["next_action_family"], "stage-replay-policy")
        self.assertIn("skipped-tools", by_code)
        self.assertIn("disabled", by_code)
        self.assertIn("true-miss", by_code)
        self.assertIn("holdout-only", by_code)
        self.assertIn("dependency-invalidation-blocked", by_code)
        self.assertIn("staged-policy-not-loaded", by_code)
        self.assertTrue(ladder_payload["privacy"]["metadata_only"])
        self.assertFalse(ladder_payload["privacy"]["raw_prompts_included"])
        self.assertFalse(ladder_payload["privacy"]["request_ids_included"])
        self.assertFalse(ladder_payload["privacy"]["session_ids_included"])
        self.assertFalse(ladder_payload["privacy"]["cache_keys_included"])
        rendered = json.dumps(ladder_payload, sort_keys=True)
        self.assertNotIn(secret_prompt, rendered)
        self.assertNotIn(secret_session, rendered)
        self.assertNotIn(secret_cache_key, rendered)
        self.assertNotIn(secret_path, rendered)

    def test_cache_zero_hit_blocker_ladder_reports_bounded_recent_window(self):
        for index in range(1005):
            server.store.log_call(
                id=f"cache-blocker-bounded-{index}",
                created_at=f"2026-06-10T00:{index // 60:02d}:{index % 60:02d}+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=0.0,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"has_tools": False, "category": "chat"}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="bounded-cache-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["cache_zero_hit_blocker_ladder"]["summary"]

        self.assertTrue(summary["bounded_recent_window"])
        self.assertEqual(summary["scan_limit"], 1000)
        self.assertEqual(summary["scanned_rows"], 1000)
        self.assertEqual(summary["available_rows"], 1000)

    def test_cache_effectiveness_endpoint_reports_local_cache_without_raw_leakage(self):
        secret_cache_key = "secret-cache-key-should-not-render"
        secret_prompt = "raw local cache prompt must not leak"
        secret_response = "raw local cache response must not leak"
        server.store.set_cache(secret_cache_key, "claude-sonnet-4-6", 100, {"content": secret_response})
        server.store.set_semantic_cache(
            "secret-semantic-key-should-not-render",
            "claude-sonnet-4-6",
            [0.1, 0.2, 0.3],
            {"content": "raw semantic response must not leak"},
            100,
        )
        rows = [
            {"cache": {"status": "miss", "reason": "exact-miss", "exact_enabled": True}, "stream": 0, "hit": 0},
            {"cache": {"status": "hit", "reason": "exact-match", "hit_type": "exact", "exact_enabled": True}, "stream": 0, "hit": 1},
            {"cache": {"status": "hit", "reason": "semantic-match", "hit_type": "semantic"}, "stream": 0, "hit": 0},
            {"cache": {"status": "skipped", "reason": "streaming", "exact_enabled": False}, "stream": 1, "hit": 0},
            {"cache": {"status": "skipped", "reason": "tools-disabled", "exact_enabled": False}, "stream": 0, "hit": 0},
            {"cache": {"status": "miss", "reason": "file-dependency-changed", "invalidation_reason": "dependency-changed"}, "stream": 0, "hit": 0},
        ]
        for index, row in enumerate(rows):
            server.store.log_call(
                id=f"cache-effectiveness-{index}",
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=row["stream"],
                cache_hit=row["hit"],
                status_code=200,
                latency_ms=1,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.002,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 400}),
                cache_json=stable_json(row["cache"]),
                error=None,
                request_json=stable_json({"messages": [{"content": secret_prompt}], "cache_key": secret_cache_key}),
                response_json=stable_json({"content": secret_response}),
                session_id="secret-cache-session",
                category="chat",
                cache_creation_input_tokens=20,
                cache_read_input_tokens=80,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_cache_effectiveness(server.store, scan_limit=20))
        summary = result["summary"]

        self.assertEqual(result["schema"], "agentflow.cache_smoke_diagnostic.v1")
        self.assertEqual(summary["exact_cache_rows"], 1)
        self.assertEqual(summary["semantic_cache_rows"], 1)
        self.assertEqual(summary["exact_lookup_count"], 2)
        self.assertEqual(summary["exact_hit_count"], 1)
        self.assertEqual(summary["semantic_hit_count"], 1)
        self.assertEqual(summary["exact_miss_count"], 1)
        self.assertEqual(summary["skip_streaming_count"], 1)
        self.assertEqual(summary["tools_disabled_skip_count"], 1)
        self.assertEqual(summary["file_dependency_blocked_count"], 1)
        self.assertIn("cache_keys_included", result["privacy"])

        client = TestClient(
            create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.anthropic.com",
                limiter_status=lambda: [],
                limiter_config={},
            )
        )
        endpoint = client.get("/agentflow/stats/cache-effectiveness?scan_limit=20")
        full = client.get("/agentflow/stats/full")
        html = client.get("/agentflow/dashboard").text

        self.assertEqual(endpoint.status_code, 200)
        self.assertEqual(endpoint.json()["summary"]["exact_hit_count"], 1)
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.json()["cache_effectiveness"]["summary"]["semantic_hit_count"], 1)
        self.assertIn("Local AgentFlow cache replay", html)
        self.assertIn("Provider prompt-cache discount", html)
        self.assertIn("local-cache-effectiveness-tbody", html)
        self.assertIn("provider-prompt-cache-tbody", html)
        rendered = json.dumps(endpoint.json(), sort_keys=True) + json.dumps(full.json(), sort_keys=True) + html
        for forbidden in (
            secret_cache_key,
            "secret-semantic-key-should-not-render",
            secret_prompt,
            secret_response,
            "raw semantic response must not leak",
            "secret-cache-session",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(endpoint.json()["privacy"]["cache_keys_included"])
        self.assertFalse(endpoint.json()["privacy"]["raw_request_bodies_included"])

    def test_pattern_decision_breakdown_reports_outcome_error_and_savings_by_rule(self):
        applied_hash = "sha256:" + "a" * 64
        rows = [
            {
                "status_code": 200,
                "cost_est_usd": 0.001,
                "cost_baseline_usd": 0.004,
                "crunch": {
                    "changed": True,
                    "policy_source": "managed-recommended",
                    "pattern_rules": {
                        "configured_count": 1,
                        "policy_source": "managed-recommended",
                        "before_chars": 4000,
                        "after_chars": 2800,
                        "rules": [
                            {
                                "rule_id": "pattern-apply",
                                "candidate_id": "candidate-apply",
                                "policy_source": "managed-recommended",
                                "matched_hashes": [applied_hash],
                                "applied_count": 1,
                                "saved_chars": 1200,
                            }
                        ],
                    },
                },
                "cache": {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            },
            {
                "status_code": 200,
                "cost_est_usd": 0.002,
                "cost_baseline_usd": 0.002,
                "crunch": {
                    "changed": False,
                    "pattern_rules": {
                        "configured_count": 1,
                        "policy_source": "managed-recommended",
                        "rules": [
                            {
                                "rule_id": "pattern-skip",
                                "policy_source": "managed-recommended",
                                "applied_count": 0,
                                "saved_chars": 0,
                                "skip_reasons": [{"reason": "min-repeated-count-not-met", "count": 1}],
                            }
                        ],
                    },
                },
                "cache": {"status": "skipped", "reason": "cache-disabled", "policy_source": "local-default"},
            },
            {
                "status_code": 400,
                "cost_est_usd": 0.0,
                "cost_baseline_usd": 0.0,
                "crunch": {
                    "changed": True,
                    "pattern_rules": {
                        "configured_count": 1,
                        "policy_source": "managed-recommended",
                        "rules": [
                            {
                                "rule_id": "pattern-error",
                                "policy_source": "managed-recommended",
                                "matched_hashes": ["sha256:" + "b" * 64],
                                "applied_count": 1,
                                "saved_chars": 800,
                            }
                        ],
                    },
                },
                "cache": {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            },
        ]
        for row in rows:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                stream=0,
                cache_hit=0,
                status_code=row["status_code"],
                latency_ms=1,
                input_tokens_est=1000,
                output_tokens_est=10,
                actual_input_tokens=1000,
                actual_output_tokens=10,
                cost_est_usd=row["cost_est_usd"],
                cost_baseline_usd=row["cost_baseline_usd"],
                crunch_json=stable_json(row["crunch"]),
                routing_json=stable_json({"category": "tool-result", "has_tools": True}),
                cache_json=stable_json(row["cache"]),
                error="upstream error raw body" if row["status_code"] >= 400 else None,
                request_json=None,
                response_json=None,
                session_id="session-patterns",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        rows_by_rule = {
            (row["decision_type"], row["rule_id"], row["outcome"]): row
            for row in result["pattern_decision_breakdown"]
        }

        applied = rows_by_rule[("crunch", "pattern-apply", "applied")]
        self.assertEqual(applied["candidate_id"], "candidate-apply")
        self.assertEqual(applied["pattern_hash"], applied_hash)
        self.assertEqual(applied["saved_chars"], 1200)
        self.assertEqual(applied["tokens_saved_est"], 300)
        self.assertEqual(applied["error_count"], 0)
        self.assertGreater(applied["estimated_cost_savings_usd"], 0)
        self.assertEqual(rows_by_rule[("crunch", "pattern-skip", "skipped")]["reason"], "min-repeated-count-not-met")
        self.assertEqual(rows_by_rule[("crunch", "pattern-error", "errored")]["error_count"], 1)
        cache_outcomes = {
            row["outcome"]
            for row in result["pattern_decision_breakdown"]
            if row["decision_type"] == "cache"
        }
        self.assertTrue({"skipped", "bypassed", "errored"}.issubset(cache_outcomes))
        self.assertEqual(result["today_pattern_decision_breakdown"], result["pattern_decision_breakdown"])
        self.assertNotIn("upstream error raw body", json.dumps(result["pattern_decision_breakdown"]))

    def test_managed_pattern_rollups_aggregate_canary_cohorts_without_raw_leakage(self):
        crunch_hash = "sha256:" + "c" * 64
        cache_hash = "sha256:" + "d" * 64
        rollback_hash = "sha256:" + "e" * 64
        codex_hash = "sha256:" + "f" * 64

        server.store.log_call(
            id="provider-canary-applied",
            created_at="2026-06-08T10:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=800,
            input_tokens_est=1000,
            output_tokens_est=100,
            actual_input_tokens=1000,
            actual_output_tokens=100,
            cost_est_usd=0.004,
            cost_baseline_usd=0.007,
            crunch_json=stable_json({
                "changed": True,
                "policy_source": "managed-recommended",
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "crunch-rule",
                            "candidate_id": "crunch-candidate",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [crunch_hash],
                            "applied_count": 1,
                            "saved_chars": 1200,
                            "canary": {
                                "schema": "agentflow.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": True,
                                "status": "applied",
                                "cohort": "canary_applied",
                                "fraction": 0.1,
                                "unit": "request_fingerprint",
                                "cohort_key_hash": "sha256:" + "1" * 64,
                            },
                        }
                    ],
                },
            }),
            routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-result"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=stable_json({"messages": [{"content": "raw prompt must stay local"}], "cache_key": "cache-key-secret"}),
            response_json=stable_json({"content": "raw response must stay local"}),
            session_id="session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id="provider-canary-holdout",
            created_at="2026-06-08T10:01:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=1200,
            input_tokens_est=900,
            output_tokens_est=80,
            actual_input_tokens=900,
            actual_output_tokens=80,
            cost_est_usd=0.003,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-result"}),
            cache_json=stable_json({
                "status": "skipped",
                "reason": "tools-disabled",
                "policy_source": "local-default",
                "pattern_rules": {
                    "configured_count": 1,
                    "skip_reasons": [
                        {
                            "rule_id": "cache-rule",
                            "candidate_id": "cache-candidate",
                            "policy_source": "managed-recommended",
                            "reason": "canary_holdout",
                            "matched_hashes": [cache_hash],
                            "canary": {
                                "schema": "agentflow.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": False,
                                "status": "holdout",
                                "cohort": "canary_holdout",
                                "fraction": 0.1,
                                "unit": "request_fingerprint",
                                "cohort_key_hash": "sha256:" + "2" * 64,
                            },
                        }
                    ],
                },
            }),
            error=None,
            request_json=stable_json({"prompt": "another raw prompt must stay local"}),
            response_json=None,
            session_id="session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id="provider-rollback-bypass",
            created_at="2026-06-08T10:02:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=5200,
            input_tokens_est=800,
            output_tokens_est=70,
            actual_input_tokens=800,
            actual_output_tokens=70,
            cost_est_usd=0.002,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"category": "chat", "workflow_phase": "chat"}),
            cache_json=stable_json({
                "status": "bypass",
                "reason": "rollback-threshold-breached",
                "policy_source": "managed-recommended",
                "pattern_rule": {
                    "rule_id": "rollback-cache-rule",
                    "candidate_id": "rollback-cache-candidate",
                    "policy_source": "managed-recommended",
                    "matched_hashes": [rollback_hash],
                },
            }),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-secret",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id="provider-canary-error",
            created_at="2026-06-08T10:03:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=0,
            cache_hit=0,
            status_code=500,
            latency_ms=20000,
            input_tokens_est=1000,
            output_tokens_est=0,
            actual_input_tokens=1000,
            actual_output_tokens=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({
                "changed": True,
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "crunch-rule",
                            "candidate_id": "crunch-candidate",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [crunch_hash],
                            "applied_count": 1,
                            "saved_chars": 400,
                            "canary": {
                                "schema": "agentflow.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": True,
                                "status": "applied",
                                "cohort": "canary_applied",
                                "fraction": 0.1,
                                "unit": "request_fingerprint",
                            },
                        }
                    ],
                },
            }),
            routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-result"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error="upstream raw error body must stay local",
            request_json=None,
            response_json=None,
            session_id="session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_codex_app_event(
            id="codex-start",
            created_at="2026-06-08T10:04:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="req-secret",
            thread_id="thread-secret",
            message_chars=40,
            params_chars=500,
            input_items=1,
            input_text_chars=2400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="session-secret",
            routing_json=stable_json({"category": "summary", "workflow_phase": "summary", "requested_model": "gpt-5-codex"}),
            crunch_json=stable_json({"status": "skipped", "reason": "not-needed"}),
            cache_json=stable_json({
                "status": "hit",
                "reason": "exact-match",
                "hit_type": "exact",
                "policy_source": "managed-recommended",
                "pattern_rule": {
                    "rule_id": "codex-cache-rule",
                    "candidate_id": "codex-cache-candidate",
                    "policy_source": "managed-recommended",
                    "matched_hashes": [codex_hash],
                    "canary": {
                        "schema": "agentflow.pattern_canary_decision.v1",
                        "enabled": True,
                        "selected": True,
                        "status": "applied",
                        "cohort": "canary_applied",
                        "fraction": 0.2,
                        "unit": "request_fingerprint",
                    },
                },
            }),
        )
        server.store.log_codex_app_event(
            id="codex-end",
            created_at="2026-06-08T10:04:01+00:00",
            direction="server_to_client",
            method="turn/completed",
            request_id="req-secret",
            thread_id="thread-secret",
            message_chars=80,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=400,
            error_code=None,
            error_message=None,
            latency_ms=900,
            session_id="session-secret",
            metadata_json=stable_json({"transcript": "raw codex transcript must stay local"}),
        )

        result = asyncio.run(stats_views.stats_managed_pattern_rollups(server.store, limit=20, min_samples=2))
        cohorts = {
            (row["policy_section"], row["candidate_id"], row["canary_cohort"]): row
            for row in result["cohorts"]
        }

        applied = cohorts[("crunch", "crunch-candidate", "canary_applied")]
        self.assertEqual(applied["sample_count"], 2)
        self.assertEqual(applied["error_count"], 1)
        self.assertEqual(applied["success_count"], 1)
        self.assertEqual(applied["tokens_saved_est"], 400)
        self.assertTrue(applied["minimum_sample_readiness"]["ready"])
        self.assertEqual(applied["pattern_hash"], crunch_hash)
        self.assertIn({"value": "gte_15s", "count": 1}, applied["latency_buckets"])
        self.assertIn({"value": "5xx", "count": 1}, applied["status_code_counts"])

        holdout = cohorts[("cache", "cache-candidate", "canary_holdout")]
        self.assertEqual(holdout["holdout_count"], 1)
        self.assertFalse(holdout["minimum_sample_readiness"]["ready"])
        self.assertEqual(holdout["minimum_sample_readiness"]["remaining"], 1)

        rollback = cohorts[("cache", "rollback-cache-candidate", "non_canary")]
        self.assertEqual(rollback["bypassed_count"], 1)
        self.assertIn({"value": "rollback-threshold-breached", "count": 1}, rollback["local_bypass_reasons"])
        self.assertIn({"value": "rolled_back", "count": 1}, rollback["lifecycle_counts"])

        codex = cohorts[("cache", "codex-cache-candidate", "canary_applied")]
        self.assertEqual(codex["source_surface"], "codex_turn")
        self.assertEqual(codex["app_family"], "codex")
        self.assertEqual(codex["success_count"], 1)
        self.assertGreater(codex["estimated_cost_savings_usd"], 0)

        self.assertEqual(result["summary"]["provider_rows_considered"], 4)
        self.assertEqual(result["summary"]["codex_turn_rows_considered"], 1)
        self.assertEqual(result["summary"]["rolled_back_events"], 1)
        self.assertTrue(result["privacy"]["aggregate_only"])
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw prompt must stay local", encoded)
        self.assertNotIn("raw response must stay local", encoded)
        self.assertNotIn("upstream raw error body", encoded)
        self.assertNotIn("raw codex transcript", encoded)
        self.assertNotIn("session-secret", encoded)
        self.assertNotIn("req-secret", encoded)
        self.assertNotIn("thread-secret", encoded)
        self.assertNotIn("cache-key-secret", encoded)

    def test_cache_decision_breakdown_infers_legacy_null_cache_rows(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:01:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=1,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-hit-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:02:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=0,
            cache_hit=0,
            status_code=400,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-error-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:03:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-unknown-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-today",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"hit_type": "skip-streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-partial",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        all_time = {
            (row["status"], row["reason"], row["policy_source"]): row["count"]
            for row in result["cache_decision_breakdown"]
        }
        today = {
            (row["status"], row["reason"]): row["count"]
            for row in result["today_cache_decision_breakdown"]
        }

        self.assertEqual(all_time[("skipped", "legacy-streaming", "legacy-inferred")], 1)
        self.assertEqual(all_time[("skipped", "legacy-streaming", "local-default")], 1)
        self.assertEqual(all_time[("hit", "legacy-cache-hit", "legacy-inferred")], 1)
        self.assertEqual(all_time[("skipped", "legacy-upstream-error", "legacy-inferred")], 1)
        self.assertEqual(all_time[("missing", "legacy-unknown", "legacy-inferred")], 1)
        self.assertEqual(today[("skipped", "streaming")], 1)
        self.assertEqual(today[("skipped", "legacy-streaming")], 1)
        json.dumps(result["today_cache_decision_breakdown"])

    def test_codex_cache_breakdown_merges_legacy_and_canonical_source_surfaces(self):
        for surface in ("codex_app_turn", "codex_turn"):
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-{surface}",
                thread_id="thread-codex-surface",
                message_chars=100,
                params_chars=50,
                input_items=1,
                input_text_chars=40,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-surface-session",
                cache_json=stable_json({
                    "status": "hit",
                    "reason": "exact-match",
                    "hit_type": "exact",
                    "policy_source": "local-default",
                    "surface": surface,
                }),
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        cache_rows = {
            (row["source_surface"], row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in result["today_cache_decision_breakdown"]
        }

        self.assertEqual(cache_rows[("codex_turn", "hit", "exact-match", "exact")], 2)
        self.assertNotIn(("codex_app_turn", "hit", "exact-match", "exact"), cache_rows)

    def test_codex_cache_breakdown_reports_canary_outcome_buckets_metadata_only(self):
        secret = "secret summary prompt must not appear in reports"
        rows = [
            {"status": "skipped", "reason": "codex-app-cache-disabled", "outcome_bucket": "disabled"},
            {"status": "holdout", "reason": "codex-app-cache-canary-holdout", "outcome_bucket": "holdout"},
            {"status": "unsafe-skip", "reason": "unsafe-cached-envelope", "outcome_bucket": "unsafe-skip"},
            {"status": "miss", "reason": "exact-miss", "outcome_bucket": "miss"},
            {"status": "hit", "reason": "exact-match", "hit_type": "exact", "outcome_bucket": "hit"},
            {"status": "miss", "reason": "dependency-changed", "outcome_bucket": "invalidated"},
            {"status": "skipped", "reason": "file-dependency-missing", "outcome_bucket": "stale-risk"},
        ]
        for index, cache_json in enumerate(rows):
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-codex-cache-bucket-{index}",
                thread_id="thread-codex-cache-buckets",
                message_chars=100 + index,
                params_chars=80,
                input_items=1,
                input_text_chars=len(secret),
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-cache-bucket-session-secret",
                cache_json=stable_json({
                    **cache_json,
                    "policy_source": "local-default",
                    "surface": "codex_turn",
                    "cache_key_included": False,
                }),
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        buckets = {
            row["outcome_bucket"]
            for row in result["today_cache_decision_breakdown"]
            if row["source_surface"] == "codex_turn"
        }

        self.assertTrue({
            "disabled",
            "holdout",
            "unsafe-skip",
            "miss",
            "hit",
            "invalidated",
            "stale-risk",
        }.issubset(buckets))
        rendered = json.dumps(result["today_cache_decision_breakdown"], sort_keys=True)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("codex-cache-bucket-session-secret", rendered)
        self.assertNotIn("cache_key", rendered)

    def test_cache_replayability_report_groups_repeated_skipped_shapes_and_blockers(self):
        def log_provider_call(
            *,
            cache_json,
            routing_json,
            session_id,
            stream,
            category,
            cost,
            request_json=None,
            crunch_json=None,
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=stream,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json(crunch_json or {"changed": False}),
                routing_json=stable_json(routing_json),
                cache_json=stable_json(cache_json),
                error=None,
                request_json=request_json,
                response_json=None,
                session_id=session_id,
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        def cacheability_crunch(
            *,
            bucket,
            static=False,
            current_state=False,
            user_specific=False,
            exact_candidate=False,
        ):
            return {
                "changed": False,
                "pattern_modules": {
                    "server_features": {
                        "features": [
                            {
                                "family": "cacheability",
                                "features": {
                                    "cacheability_bucket": bucket,
                                    "deterministic_answer_likelihood_bucket": "high" if bucket == "high" else "low",
                                    "static_information_hint": static,
                                    "time_sensitive_hint": current_state,
                                    "user_specific_hint": user_specific,
                                    "exact_cache_candidate_hint": exact_candidate,
                                    "cache_preserved_by_default_reason": "none" if exact_candidate else "current-state",
                                },
                            }
                        ],
                    },
                },
            }

        streaming_cache = {
            "status": "skipped",
            "reason": "streaming",
            "policy_source": "local-default",
            "semantic_enabled": False,
            "tool_cache_enabled": False,
            "file_watch_enabled": False,
        }
        streaming_routing = {"text_chars": 12_000, "category": "chat", "has_tools": False}
        log_provider_call(
            cache_json=streaming_cache,
            routing_json=streaming_routing,
            session_id="session-stream-a",
            stream=1,
            category="chat",
            cost=0.01,
            crunch_json=cacheability_crunch(bucket="high", static=True, exact_candidate=True),
            request_json=stable_json({"messages": [{"content": "raw-secret-should-not-leak"}]}),
        )
        log_provider_call(
            cache_json=streaming_cache,
            routing_json=streaming_routing,
            session_id="session-stream-a",
            stream=1,
            category="chat",
            cost=0.02,
            crunch_json=cacheability_crunch(bucket="high", static=True, exact_candidate=True),
        )

        tool_cache = {
            "status": "skipped",
            "reason": "tools-disabled",
            "policy_source": "local-default",
            "semantic_enabled": False,
            "tool_cache_enabled": False,
            "file_watch_enabled": False,
        }
        tool_routing = {"text_chars": 24_000, "category": "tool-result", "has_tools": True}
        log_provider_call(
            cache_json=tool_cache,
            routing_json=tool_routing,
            session_id="session-tool-a",
            stream=0,
            category="tool-result",
            cost=0.03,
        )
        log_provider_call(
            cache_json=tool_cache,
            routing_json=tool_routing,
            session_id="session-tool-b",
            stream=0,
            category="tool-result",
            cost=0.04,
        )

        static_cache = {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}
        static_routing = {"text_chars": 600, "category": "chat", "has_tools": False}
        log_provider_call(
            cache_json=static_cache,
            routing_json=static_routing,
            session_id="session-static",
            stream=0,
            category="chat",
            cost=0.02,
            crunch_json=cacheability_crunch(bucket="high", static=True, exact_candidate=True),
        )
        log_provider_call(
            cache_json=static_cache,
            routing_json=static_routing,
            session_id="session-static",
            stream=0,
            category="chat",
            cost=0.03,
            crunch_json=cacheability_crunch(bucket="high", static=True, exact_candidate=True),
        )

        log_provider_call(
            cache_json={"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            routing_json={"text_chars": 800, "category": "chat", "has_tools": False},
            session_id="session-current",
            stream=0,
            category="chat",
            cost=0.006,
            crunch_json=cacheability_crunch(
                bucket="low",
                current_state=True,
                user_specific=True,
                exact_candidate=False,
            ),
        )

        log_provider_call(
            cache_json={"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            routing_json={"text_chars": 400, "category": "short-completion", "has_tools": False},
            session_id="session-one-off",
            stream=0,
            category="short-completion",
            cost=0.005,
        )

        result = asyncio.run(stats_views.stats_cache_replayability(server.store, limit=10))
        groups = {(row["cache_reason"], row["category"], row["cacheability_bucket"]): row for row in result["groups"]}

        self.assertEqual(result["schema"], "agentflow.cache_replayability.v1")
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertEqual(result["summary"]["repeated_shape_groups"], 3)
        self.assertTrue(result["summary"]["repeated_shape_exists_but_cache_is_unsafe"])
        self.assertAlmostEqual(result["summary"]["projected_repeated_call_cost_usd"], 0.075)
        self.assertEqual(groups[("streaming", "chat", "high")]["count"], 2)
        self.assertEqual(groups[("streaming", "chat", "high")]["replay_candidate_class"], "streaming-non-tool-exact-candidate")
        self.assertIn("streaming", groups[("streaming", "chat", "high")]["replayability_blockers"])
        self.assertEqual(groups[("tools-disabled", "tool-result", "unknown")]["count"], 2)
        self.assertEqual(groups[("tools-disabled", "tool-result", "unknown")]["sessions"], 2)
        self.assertEqual(groups[("tools-disabled", "tool-result", "unknown")]["replay_candidate_class"], "blocked-tool-result-invalidation")
        self.assertIn("tool-call-disabled", groups[("tools-disabled", "tool-result", "unknown")]["replayability_blockers"])
        self.assertIn("file-dependency-missing", groups[("tools-disabled", "tool-result", "unknown")]["replayability_blockers"])
        self.assertIn("session-context-changed", groups[("tools-disabled", "tool-result", "unknown")]["replayability_blockers"])
        self.assertEqual(groups[("exact-miss", "chat", "high")]["replay_candidate_class"], "replay-safe-exact-candidate")
        self.assertEqual(groups[("exact-miss", "chat", "low")]["replay_candidate_class"], "blocked-low-cacheability")
        self.assertIn("current-state", groups[("exact-miss", "chat", "low")]["replayability_blockers"])
        self.assertIn("user-specific", groups[("exact-miss", "chat", "low")]["replayability_blockers"])
        self.assertIn("true-one-off-miss", groups[("exact-miss", "short-completion", "unknown")]["replayability_blockers"])
        self.assertNotIn("raw-secret-should-not-leak", json.dumps(result))

        by_blocker = {row["blocker"]: row["calls"] for row in result["blocker_breakdown"]}
        self.assertEqual(by_blocker["streaming"], 2)
        self.assertEqual(by_blocker["tool-call-disabled"], 2)
        self.assertEqual(by_blocker["file-dependency-missing"], 2)
        self.assertEqual(by_blocker["true-one-off-miss"], 1)
        self.assertEqual(by_blocker["current-state"], 1)

        burn_down = result["blocker_burn_down"]
        self.assertEqual(result["summary"]["blocker_burn_down_rows"], 3)
        self.assertEqual(burn_down[0]["next_action_family"], "tool_call_safety")
        self.assertEqual(burn_down[0]["source_surface"], "anthropic_messages")
        self.assertEqual(burn_down[0]["workflow_phase"], "tool-result")
        self.assertEqual(burn_down[0]["category"], "tool-result")
        self.assertEqual(burn_down[0]["calls"], 2)
        self.assertEqual(burn_down[0]["shape_groups"], 1)
        self.assertEqual(burn_down[0]["projected_cost_bucket"], "1c_5c")
        self.assertAlmostEqual(burn_down[0]["projected_repeated_call_cost_usd"], 0.035)
        self.assertIn("tool-call-disabled", burn_down[0]["blockers"])
        self.assertIn("file-dependency-missing", burn_down[0]["blockers"])
        self.assertFalse(burn_down[0]["raw_prompts_included"])
        self.assertFalse(burn_down[0]["file_paths_included"])
        self.assertEqual(result["summary"]["top_blocker_burn_down_next_action_family"], "tool_call_safety")
        self.assertAlmostEqual(result["summary"]["top_blocker_burn_down_projected_cost_usd"], 0.035)
        action_by_family = {row["next_action_family"]: row for row in burn_down}
        self.assertEqual(action_by_family["canary_policy_loading"]["blockers"], ["none"])
        self.assertAlmostEqual(action_by_family["canary_policy_loading"]["projected_repeated_call_cost_usd"], 0.025)
        self.assertEqual(action_by_family["streaming_replay"]["projected_cost_bucket"], "1c_5c")
        self.assertAlmostEqual(action_by_family["streaming_replay"]["projected_repeated_call_cost_usd"], 0.015)

    def test_cache_replayability_burn_down_empty_dataset(self):
        result = asyncio.run(stats_views.stats_cache_replayability(server.store, limit=10))

        self.assertEqual(result["schema"], "agentflow.cache_replayability.v1")
        self.assertEqual(result["summary"]["candidate_rows"], 0)
        self.assertEqual(result["summary"]["blocker_burn_down_rows"], 0)
        self.assertEqual(result["summary"]["top_blocker_burn_down_projected_cost_usd"], 0.0)
        self.assertIsNone(result["summary"]["top_blocker_burn_down_next_action_family"])
        self.assertEqual(result["blocker_burn_down"], [])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])

    def test_cache_replayability_endpoint_and_dashboard_are_read_only_metadata(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.01,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"text_chars": 12_000, "category": "chat", "has_tools": False}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=stable_json({"messages": [{"content": "private request body"}]}),
            response_json=None,
            session_id="session-api",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )

        with TestClient(app) as client:
            response = client.get("/agentflow/stats/cache-replayability?limit=5")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["schema"], "agentflow.cache_replayability.v1")
            self.assertFalse(data["privacy"]["raw_prompts_included"])
            self.assertNotIn("private request body", json.dumps(data))
            html = client.get("/agentflow/dashboard").text
            self.assertIn("Cache blocker burn-down", html)
            self.assertIn("cache-blocker-burn-down-tbody", html)
            self.assertIn("Skipped cache replayability", html)
            self.assertIn("cache-replayability-tbody", html)

    def test_cache_replayability_dashboard_api_sanitizes_raw_like_labels(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=100,
            output_tokens_est=1,
            actual_input_tokens=100,
            actual_output_tokens=1,
            cost_est_usd=0.01,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "text_chars": 12_000,
                "category": "raw replay prompt /tmp/replay-label-secret.py",
                "workflow_phase": "raw replay response message",
                "has_tools": False,
            }),
            cache_json=stable_json({
                "status": "miss",
                "reason": "cache-key-secret req-replay-secret prompt",
                "policy_source": "managed-recommended",
                "rule_id": "raw rule id cache-key-secret",
                "candidate_id": "raw candidate id request-id-secret",
                "session_memory_hints": {
                    "dry_run_replay_proposal": {
                        "schema": "agentflow.session_memory_cache_replay_proposal.v1",
                        "status": "raw status prompt",
                        "reason": "cache-key-secret reason",
                        "proposal_id": "raw proposal id must not leak",
                        "proposal_fingerprint": "sha256:" + "a" * 16,
                        "rule_id": "raw rule id request-id-secret",
                        "policy_source": "managed-recommended",
                        "phase": "raw phase prompt",
                        "category": "raw category /tmp/proposal-secret.py",
                        "blockers": ["cache-key-secret blocker"],
                        "review_steps": ["review metadata-only session plateau shape"],
                        "privacy": {"metadata_only": True},
                    }
                },
            }),
            error=None,
            request_json=stable_json({"messages": [{"content": "private replayability prompt"}]}),
            response_json=stable_json({"content": [{"text": "private replayability response"}]}),
            session_id="raw-replayability-session-secret",
            category="raw replay prompt /tmp/replay-label-secret.py",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            payload = client.get("/agentflow/stats/cache-replayability?limit=10").json()
            readiness = client.get("/agentflow/stats/cache-replay-readiness?limit=10").json()
            html = client.get("/agentflow/dashboard").text

        rendered = json.dumps({"payload": payload, "readiness": readiness}, sort_keys=True) + html
        for forbidden in (
            "raw replay prompt",
            "/tmp/replay-label-secret.py",
            "raw replay response",
            "cache-key-secret",
            "req-replay-secret",
            "raw rule id",
            "raw candidate id",
            "raw proposal id",
            "/tmp/proposal-secret.py",
            "private replayability prompt",
            "private replayability response",
            "raw-replayability-session-secret",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["file_paths_included"])
        self.assertEqual(payload["groups"][0]["category"], "unknown")
        self.assertEqual(payload["groups"][0]["cache_reason"], "unknown")

    def test_cache_replayability_reports_session_memory_dry_run_proposals_metadata_only(self):
        from agentflow_proxy.session_memory_hints import build_session_memory_optimization_hints

        def log_plateau_call(suffix, *, session_id, cache_json=None, text_chars=40_000, cost=0.02):
            server.store.log_call(
                id=f"session-memory-proposal-{suffix}",
                created_at=f"2026-06-10T11:{int(suffix):02d}:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=text_chars // 4,
                output_tokens_est=50,
                actual_input_tokens=text_chars // 4,
                actual_output_tokens=50,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "text_chars": text_chars,
                    "category": "summary",
                    "workflow_phase": "summary",
                    "has_tools": False,
                }),
                cache_json=stable_json(cache_json or {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
                error=None,
                request_json=stable_json({"messages": [{"content": "private session memory prompt /tmp/proposal.py"}]}),
                response_json=stable_json({"content": [{"text": "private session memory response"}]}),
                session_id=session_id,
                category="summary",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        crunch_policy = {
            "session_memory_hints": {
                "enabled": True,
                "rule_id": "stats-session-crunch",
                "crunch_profile": "stats-profile",
                "min_call_count": 4,
                "min_plateau_pairs": 3,
                "min_text_chars": 8000,
                "allowed_phases": ["summary"],
                "projected_savings_ratio": 0.20,
            }
        }
        cache_policy = {
            "session_memory_hints": {
                "enabled": True,
                "rule_id": "stats-session-cache",
                "min_call_count": 4,
                "min_plateau_pairs": 3,
                "min_text_chars": 8000,
                "allowed_phases": ["summary"],
                "require_safe_invalidation": False,
                "require_reviewed_pattern_rule": False,
                "allow_tool_calls": False,
                "allow_streaming_replay": False,
                "projected_savings_ratio": 0.20,
            }
        }
        default_blocking_cache_policy = {
            "session_memory_hints": {
                **cache_policy["session_memory_hints"],
                "rule_id": "stats-session-cache-default-blocked",
                "require_safe_invalidation": True,
                "require_reviewed_pattern_rule": True,
            }
        }

        for index, chars in enumerate((40_000, 40_500, 39_800, 40_100), start=1):
            log_plateau_call(index, session_id="raw-session-memory-proposal-secret", text_chars=chars)
        eligible_hints = build_session_memory_optimization_hints(
            store_obj=server.store,
            session_id="raw-session-memory-proposal-secret",
            stream=False,
            has_tool_blocks=False,
            category="summary",
            text_chars=41_000,
            routing_meta={"workflow_phase": "summary"},
            crunch_policy=crunch_policy,
            crunch_policy_source="local-manual",
            crunch_rule_path="/tmp/crunch_rules.yaml",
            cache_policy=cache_policy,
            cache_policy_source="local-manual",
            cache_rule_path="/tmp/cache_rules.yaml",
            safe_invalidation_evidence=False,
            reviewed_cache_pattern_rule=False,
        )
        blocked_hints = build_session_memory_optimization_hints(
            store_obj=server.store,
            session_id="raw-session-memory-proposal-secret",
            stream=False,
            has_tool_blocks=False,
            category="summary",
            text_chars=41_000,
            routing_meta={"workflow_phase": "summary"},
            crunch_policy=crunch_policy,
            crunch_policy_source="local-manual",
            crunch_rule_path="/tmp/crunch_rules.yaml",
            cache_policy=default_blocking_cache_policy,
            cache_policy_source="local-default",
            cache_rule_path="/tmp/cache_rules.yaml",
            safe_invalidation_evidence=False,
            reviewed_cache_pattern_rule=False,
        )
        log_plateau_call(
            5,
            session_id="raw-session-memory-proposal-secret",
            cache_json={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "local-default",
                "session_memory_hints": eligible_hints["cache"],
            },
            text_chars=41_000,
            cost=0.05,
        )
        log_plateau_call(
            6,
            session_id="raw-session-memory-proposal-secret",
            cache_json={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "local-default",
                "session_memory_hints": blocked_hints["cache"],
            },
            text_chars=41_200,
            cost=0.07,
        )

        result = asyncio.run(stats_views.stats_cache_replayability(server.store, limit=10))
        proposals = result["session_memory_replay_proposals"]
        eligible = next(row for row in proposals if row["status"] == "session-plateau-dry-run-eligible")
        blocked = next(row for row in proposals if row["rule_id"] == "stats-session-cache-default-blocked")

        self.assertEqual(result["summary"]["session_memory_replay_proposal_count"], 2)
        self.assertEqual(result["summary"]["session_memory_replay_eligible_count"], 1)
        self.assertEqual(eligible["rule_id"], "stats-session-cache")
        self.assertEqual(eligible["phase"], "summary")
        self.assertEqual(eligible["category"], "summary")
        self.assertFalse(eligible["mutation_applied"])
        self.assertFalse(eligible["cache_mutation"])
        self.assertFalse(eligible["policy_files_written"])
        self.assertFalse(eligible["privacy"]["raw_request_bodies_included"])
        self.assertIn("missing_invalidation_evidence", blocked["blockers"])
        self.assertIn("reviewed_pattern_rule_required", blocked["blockers"])
        self.assertTrue(blocked["blocker_families"]["safe_invalidation"])
        self.assertTrue(blocked["blocker_families"]["reviewed_pattern_rule"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            payload = client.get("/agentflow/stats/cache-replayability?limit=10").json()
            readiness = client.get("/agentflow/stats/cache-replay-readiness?limit=10").json()

        self.assertEqual(payload["session_memory_replay_proposals"][0]["schema"], "agentflow.session_memory_cache_replay_proposal.v1")
        self.assertEqual(readiness["summary"]["session_memory_replay_proposal_count"], 2)
        rendered = json.dumps({"payload": payload, "readiness": readiness}, sort_keys=True)
        for forbidden in (
            "private session memory prompt",
            "private session memory response",
            "raw-session-memory-proposal-secret",
            "/tmp/proposal.py",
            "/tmp/cache_rules.yaml",
            "/tmp/crunch_rules.yaml",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_confidence_endpoint_and_dashboard_are_read_only_metadata(self):
        base_rule = {
            "rule_id": "static-chat-cache",
            "candidate_id": "candidate-static",
            "policy_source": "managed-recommended",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": {
                "canary_enabled": True,
                "canary_fraction": 0.5,
                "canary_unit": "request_fingerprint",
            },
            "canary": {
                "enabled": True,
                "selected": True,
                "cohort": "canary_applied",
                "fraction": 0.5,
                "pattern_hashes": ["sha256:" + "a" * 64],
            },
        }

        def log_cache_row(
            *,
            cache_json,
            status_code=200,
            retry_count=0,
            latency_ms=3,
            stream=0,
            category="chat",
            has_tools=False,
            cost=0.01,
            baseline=0.02,
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=stream,
                cache_hit=1 if cache_json.get("status") == "hit" else 0,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"text_chars": 900, "category": category, "has_tools": has_tools}),
                cache_json=stable_json(cache_json),
                error=None,
                request_json=stable_json({"messages": [{"content": "private replay prompt"}]}),
                response_json=None,
                session_id="raw-session-id-should-not-leak",
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
            )

        log_cache_row(
            cache_json={
                "status": "hit",
                "reason": "exact-match",
                "hit_type": "exact",
                "policy_source": "managed-recommended",
                "estimated_saved_cost_usd": 0.012,
                "pattern_rule": base_rule,
            },
            cost=0.001,
            baseline=0.013,
            latency_ms=4,
        )
        log_cache_row(
            cache_json={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "managed-recommended",
                "pattern_rule": base_rule,
            },
        )
        holdout_rule = {
            **base_rule,
            "canary": {
                "enabled": True,
                "selected": False,
                "cohort": "canary_holdout",
                "fraction": 0.5,
                "pattern_hashes": ["sha256:" + "b" * 64],
            },
            "reason": "canary_holdout",
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "canary_holdout",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [holdout_rule]},
            },
            status_code=429,
            retry_count=2,
            latency_ms=44,
        )
        safety_rule = {
            **base_rule,
            "reason": "local-canary-safety-stop",
            "safety_stop": {
                "reason": "local-canary-safety-stop",
                "decision": "stop",
                "sample_count": 9,
                "error_rate": 0.4,
            },
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "local-canary-safety-stop",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [safety_rule]},
            },
        )
        tool_rule = {
            "rule_id": "tool-result-cache",
            "candidate_id": "candidate-tool",
            "policy_source": "managed-recommended",
            "allow_tool_calls": True,
            "safe_invalidation": True,
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "dependency-changed",
                "policy_source": "managed-recommended",
                "pattern_rule": tool_rule,
                "file_dependency_audit": {
                    "invalidation_reason": "dependency-changed",
                    "changed_path_count": 1,
                    "paths": ["/tmp/private.py"],
                    "paths_included": True,
                },
            },
            category="tool-result",
            has_tools=True,
        )
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "stale-risk-blockers",
                "policy_source": "managed-recommended",
                "pattern_rule": {
                    **base_rule,
                    "rule_id": "stale-cache",
                    "candidate_id": "candidate-stale",
                },
                "cacheability": {
                    "cacheability_bucket": "low",
                    "time_sensitive_hint": True,
                    "user_specific_hint": True,
                },
            },
        )

        result = asyncio.run(stats_views.stats_cache_replay_confidence(server.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.cache_replay_confidence.v1")
        self.assertEqual(result["summary"]["hit_rows"], 1)
        self.assertEqual(result["summary"]["miss_rows"], 1)
        self.assertEqual(result["summary"]["holdout_rows"], 1)
        self.assertEqual(result["summary"]["safety_stop_rows"], 1)
        self.assertGreaterEqual(result["summary"]["invalidation_rows"], 1)
        self.assertGreaterEqual(result["summary"]["stale_risk_blocked_rows"], 1)
        self.assertAlmostEqual(result["summary"]["estimated_saved_cost_usd"], 0.012)
        static = next(row for row in result["rules"] if row["rule_id"] == "static-chat-cache")
        self.assertEqual(static["hit_count"], 1)
        self.assertEqual(static["miss_count"], 1)
        self.assertEqual(static["holdout_count"], 1)
        self.assertEqual(static["safety_stop_count"], 1)
        self.assertEqual(static["holdout_error_count"], 1)
        self.assertEqual(static["holdout_retry_count"], 2)
        self.assertEqual(static["replayed_avg_latency_ms"], 4)
        self.assertEqual(static["holdout_avg_latency_ms"], 44)
        self.assertAlmostEqual(static["replayed_estimated_saved_cost_usd"], 0.012)
        self.assertAlmostEqual(static["holdout_estimated_saved_cost_usd"], 0.01)
        self.assertAlmostEqual(static["replayed_savings_rate_usd"], 0.012)
        self.assertAlmostEqual(static["holdout_savings_rate_usd"], 0.01)
        self.assertTrue(static["safety_stop_active"])
        self.assertEqual(static["canary"]["fraction"], 0.5)
        tool = next(row for row in result["rules"] if row["rule_id"] == "tool-result-cache")
        self.assertEqual(tool["category"], "tool-result")
        self.assertTrue(tool["has_tools"])
        self.assertIn("dependency-changed", {row["value"] for row in tool["invalidation_reasons"]})
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("private replay prompt", rendered)
        self.assertNotIn("raw-session-id-should-not-leak", rendered)
        self.assertNotIn("/tmp/private.py", rendered)
        self.assertNotIn("sha256:" + "a" * 64, rendered)
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/cache-replay-confidence?limit=10")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["summary"]["hit_rows"], 1)
            html = client.get("/agentflow/dashboard").text
            self.assertIn("Cache replay confidence", html)
            self.assertIn("cache-replay-confidence-tbody", html)
            self.assertNotIn("private replay prompt", html)

    def test_cache_replay_readiness_endpoint_and_dashboard_show_blockers_without_raw_data(self):
        from agentflow_proxy import cache as cache_module

        base_rule = {
            "rule_id": "static-chat-cache",
            "candidate_id": "candidate-static",
            "policy_source": "managed-recommended",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": {
                "canary_enabled": True,
                "canary_fraction": 0.5,
                "canary_unit": "request_fingerprint",
            },
            "canary": {
                "enabled": True,
                "selected": True,
                "cohort": "canary_applied",
                "fraction": 0.5,
                "pattern_hashes": ["sha256:" + "c" * 64],
            },
        }

        def log_cache_row(
            *,
            cache_json,
            status_code=200,
            retry_count=0,
            latency_ms=3,
            category="chat",
            has_tools=False,
            cost=0.01,
            baseline=0.02,
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=1 if cache_json.get("status") == "hit" else 0,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"text_chars": 900, "category": category, "has_tools": has_tools}),
                cache_json=stable_json(cache_json),
                error=None,
                request_json=stable_json({"messages": [{"content": "private readiness prompt /tmp/secret.py cache-key-secret"}]}),
                response_json=None,
                session_id="raw-readiness-session-id",
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
            )

        log_cache_row(
            cache_json={
                "status": "hit",
                "reason": "exact-match",
                "policy_source": "managed-recommended",
                "estimated_saved_cost_usd": 0.012,
                "pattern_rule": base_rule,
            },
            cost=0.001,
            baseline=0.013,
        )
        holdout_rule = {
            **base_rule,
            "canary": {
                "enabled": True,
                "selected": False,
                "cohort": "canary_holdout",
                "fraction": 0.5,
                "pattern_hashes": ["sha256:" + "d" * 64],
            },
            "reason": "canary_holdout",
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "canary_holdout",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [holdout_rule]},
            },
            status_code=429,
            retry_count=1,
            latency_ms=40,
        )
        tool_rule = {
            "rule_id": "tool-result-cache",
            "candidate_id": "candidate-tool",
            "policy_source": "managed-recommended",
            "allow_tool_calls": True,
            "safe_invalidation": True,
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "dependency-changed",
                "policy_source": "managed-recommended",
                "pattern_rule": tool_rule,
                "file_dependency_audit": {
                    "invalidation_reason": "dependency-changed",
                    "changed_path_count": 1,
                    "paths": ["/tmp/secret.py"],
                    "paths_included": True,
                },
            },
            category="tool-result",
            has_tools=True,
        )
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "stale-risk-blockers",
                "policy_source": "managed-recommended",
                "pattern_rule": {
                    **base_rule,
                    "rule_id": "stale-cache",
                    "candidate_id": "candidate-stale",
                },
                "cacheability": {
                    "cacheability_bucket": "low",
                    "time_sensitive_hint": True,
                    "user_specific_hint": True,
                },
            },
        )
        safety_rule = {
            **base_rule,
            "rule_id": "stopped-cache",
            "candidate_id": "candidate-stop",
            "reason": "local-canary-safety-stop",
            "safety_stop": {"reason": "local-canary-safety-stop", "sample_count": 4, "error_rate": 0.5},
        }
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "local-canary-safety-stop",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [safety_rule]},
            },
            status_code=500,
        )

        original_rules = cache_module.CACHE_PATTERN_RULES
        cache_module.CACHE_PATTERN_RULES = (
            {
                "id": "disabled-cache-rule",
                "enabled": False,
                "policy_source": "local-manual",
                "candidate_id": "candidate-disabled",
                "conditions": {"pattern_hashes": ["sha256:" + "e" * 64]},
                "action": {"type": "exact_cache_pattern", "allow_tool_calls": True, "safe_invalidation": True},
                "rollout": {"canary_enabled": True, "canary_fraction": 0.25},
            },
        )
        try:
            result = asyncio.run(stats_views.stats_cache_replay_readiness(server.store, limit=20))
        finally:
            cache_module.CACHE_PATTERN_RULES = original_rules

        self.assertEqual(result["schema"], "agentflow.cache_replay_readiness.v1")
        self.assertTrue(result["summary"]["safety_stop_active"])
        self.assertEqual(result["summary"]["ready_rule_count"], 1)
        self.assertGreaterEqual(result["summary"]["invalidation_rows"], 1)
        by_rule = {row["rule_id"]: row for row in result["rules"]}
        self.assertEqual(by_rule["static-chat-cache"]["readiness"], "ready")
        self.assertEqual(by_rule["tool-result-cache"]["readiness"], "blocked")
        self.assertEqual(by_rule["tool-result-cache"]["dependency_evidence_status"], "blocked")
        self.assertEqual(by_rule["stopped-cache"]["readiness"], "safety-stopped")
        self.assertEqual(by_rule["disabled-cache-rule"]["readiness"], "disabled")
        self.assertIn("dependency-changed", {row["value"] for row in result["invalidation_breakdown"]})
        self.assertIn("current-state", {row["value"] for row in result["stale_risk_breakdown"]})
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("private readiness prompt", rendered)
        self.assertNotIn("raw-readiness-session-id", rendered)
        self.assertNotIn("/tmp/secret.py", rendered)
        self.assertNotIn("cache-key-secret", rendered)
        self.assertNotIn("sha256:" + "c" * 64, rendered)
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["pattern_hashes_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        cache_module.CACHE_PATTERN_RULES = (
            {
                "id": "disabled-cache-rule",
                "enabled": False,
                "policy_source": "local-manual",
                "candidate_id": "candidate-disabled",
                "conditions": {"pattern_hashes": ["sha256:" + "e" * 64]},
                "action": {"type": "exact_cache_pattern", "allow_tool_calls": True, "safe_invalidation": True},
                "rollout": {"canary_enabled": True, "canary_fraction": 0.25},
            },
        )
        try:
            with TestClient(app) as client:
                response = client.get("/agentflow/stats/cache-replay-readiness?limit=20")
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["schema"], "agentflow.cache_replay_readiness.v1")
                dashboard = client.get("/agentflow/dashboard")
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn("Cache canary cohorts", dashboard.text)
                self.assertIn("cache-canary-cohorts-tbody", dashboard.text)
                self.assertIn("Cache replay activation readiness", dashboard.text)
                self.assertIn("cache-replay-readiness-tbody", dashboard.text)
                self.assertNotIn("private readiness prompt", dashboard.text)
        finally:
            cache_module.CACHE_PATTERN_RULES = original_rules

    def test_cache_replay_activation_health_shows_canary_recovery_and_blockers(self):
        healthy_rule = {
            "rule_id": "healthy-cache-replay",
            "candidate_id": "candidate-healthy-cache-replay",
            "policy_source": "managed-recommended",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": {
                "canary_enabled": True,
                "canary_fraction": 0.25,
                "holdout_fraction": 0.75,
                "canary_unit": "request_fingerprint",
            },
            "canary": {
                "enabled": True,
                "selected": True,
                "cohort": "canary_applied",
                "fraction": 0.25,
                "holdout_fraction": 0.75,
                "pattern_hashes": ["sha256:" + "f" * 64],
            },
        }
        holdout_rule = {
            **healthy_rule,
            "reason": "canary_holdout",
            "canary": {
                **healthy_rule["canary"],
                "selected": False,
                "cohort": "canary_holdout",
                "pattern_hashes": ["sha256:" + "1" * 64],
            },
        }
        blocked_rule = {
            "rule_id": "blocked-tool-cache-replay",
            "candidate_id": "candidate-blocked-cache-replay",
            "policy_source": "managed-recommended",
            "scope": "session",
            "allow_tool_calls": True,
            "safe_invalidation": True,
            "canary": {
                "enabled": True,
                "selected": True,
                "cohort": "canary_applied",
                "fraction": 0.1,
                "pattern_hashes": ["sha256:" + "2" * 64],
            },
            "provider_adoption_gate": {
                "status": "blocked",
                "blocking": True,
                "reason_codes": ["provider-adoption-regression"],
            },
        }

        def log_cache_row(
            *,
            cache_json,
            status_code=200,
            stream=0,
            category="summary",
            workflow_phase="summary",
            has_tools=False,
            cost=0.01,
            baseline=0.02,
            latency=4,
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=stream,
                cache_hit=1 if cache_json.get("status") == "hit" else 0,
                status_code=status_code,
                latency_ms=latency,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "text_chars": 12_000,
                    "category": category,
                    "workflow_phase": workflow_phase,
                    "has_tools": has_tools,
                }),
                cache_json=stable_json(cache_json),
                error=None,
                request_json=stable_json({
                    "messages": [{"content": "private activation prompt /tmp/activation-secret.py cache-key-secret"}],
                    "request_id": "req-activation-secret",
                }),
                response_json=stable_json({"content": [{"text": "private activation response"}]}),
                session_id="raw-activation-session-secret",
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        log_cache_row(
            cache_json={
                "status": "hit",
                "reason": "exact-match",
                "policy_source": "managed-recommended",
                "estimated_saved_cost_usd": 0.012,
                "projected_hits": 3,
                "projected_saved_cost_usd": 0.03,
                "pattern_rule": healthy_rule,
                "cache_replay_canary": {
                    "schema": "agentflow.cache_replay_canary_decision.v1",
                    "rule_id": "healthy-cache-replay",
                    "candidate_id": "candidate-healthy-cache-replay",
                    "policy_source": "managed-recommended",
                    "scope": "session",
                    "canary": healthy_rule["canary"],
                    "canary_cohort": "canary_applied",
                    "status": "applied",
                    "reason": "dependency-stable",
                },
                "provider_adoption_gate": {"status": "ready", "blocking": False},
            },
            cost=0.001,
            baseline=0.013,
        )
        log_cache_row(
            cache_json={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "managed-recommended",
                "projected_hits": 2,
                "projected_saved_cost_usd": 0.02,
                "pattern_rule": healthy_rule,
                "cache_replay_canary": {
                    "schema": "agentflow.cache_replay_canary_decision.v1",
                    "rule_id": "healthy-cache-replay",
                    "candidate_id": "candidate-healthy-cache-replay",
                    "policy_source": "managed-recommended",
                    "scope": "session",
                    "canary": healthy_rule["canary"],
                    "canary_cohort": "canary_applied",
                    "status": "applied",
                    "reason": "dependency-stable",
                },
                "provider_adoption_gate": {"status": "ready", "blocking": False},
            },
        )
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "canary_holdout",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [holdout_rule]},
                "cache_replay_canary": {
                    "schema": "agentflow.cache_replay_canary_decision.v1",
                    "rule_id": "healthy-cache-replay",
                    "candidate_id": "candidate-healthy-cache-replay",
                    "policy_source": "managed-recommended",
                    "scope": "session",
                    "canary": holdout_rule["canary"],
                    "canary_cohort": "canary_holdout",
                    "status": "holdout",
                    "reason": "canary_holdout",
                },
                "provider_adoption_gate": {"status": "ready", "blocking": False},
            },
            baseline=0.02,
        )
        log_cache_row(
            cache_json={
                "status": "skipped",
                "reason": "dependency-changed",
                "policy_source": "managed-recommended",
                "pattern_rule": blocked_rule,
                "cache_replay_canary": {
                    "schema": "agentflow.cache_replay_canary_decision.v1",
                    "rule_id": "blocked-tool-cache-replay",
                    "candidate_id": "candidate-blocked-cache-replay",
                    "policy_source": "managed-recommended",
                    "scope": "session",
                    "canary": blocked_rule["canary"],
                    "canary_cohort": "canary_applied",
                    "status": "invalidated",
                    "reason": "dependency-changed",
                },
                "file_dependency_audit": {
                    "invalidation_reason": "dependency-changed",
                    "changed_path_count": 1,
                    "paths": ["/tmp/activation-secret.py"],
                    "paths_included": True,
                },
                "provider_adoption_gate": blocked_rule["provider_adoption_gate"],
            },
            category="tool-result",
            workflow_phase="tool-execution",
            has_tools=True,
        )

        result = asyncio.run(stats_views.stats_cache_replay_activation_health(server.store, limit=20, scan_limit=20))
        self.assertEqual(result["schema"], "agentflow.cache_replay_activation_health.v1")
        self.assertTrue(result["read_only"])
        self.assertGreaterEqual(result["summary"]["healthy_canary_count"], 1)
        self.assertGreaterEqual(result["summary"]["blocked_or_hold_count"], 1)
        by_rule = {row["rule_id"]: row for row in result["cohorts"]}
        self.assertEqual(by_rule["healthy-cache-replay"]["state"], "widen candidate")
        self.assertEqual(by_rule["healthy-cache-replay"]["hit_count"], 1)
        self.assertEqual(by_rule["healthy-cache-replay"]["holdout_count"], 1)
        self.assertEqual(by_rule["healthy-cache-replay"]["projected_hits"], 5)
        self.assertEqual(by_rule["blocked-tool-cache-replay"]["state"], "hold")
        self.assertEqual(by_rule["blocked-tool-cache-replay"]["provider_adoption_gate"]["status"], "blocked")
        self.assertIn("provider-adoption-regression", by_rule["blocked-tool-cache-replay"]["reason_codes"])
        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "private activation prompt",
            "private activation response",
            "/tmp/activation-secret.py",
            "cache-key-secret",
            "req-activation-secret",
            "raw-activation-session-secret",
            "sha256:" + "f" * 64,
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["pattern_hashes_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/cache-replay-activation-health?limit=20&scan_limit=20")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["summary"]["widen_candidate_count"], 1)
            dashboard = client.get("/agentflow/dashboard")
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Cache replay activation health", dashboard.text)
            self.assertIn("cache-replay-activation-health-tbody", dashboard.text)
            self.assertNotIn("private activation prompt", dashboard.text)

    def test_streaming_cache_hit_recovery_reports_store_missing_without_raw_data(self):
        rule = {
            "rule_id": "streaming-cache-rule-store-missing",
            "candidate_id": "candidate-store-missing",
            "policy_id": "policy-store-missing",
            "policy_source": "managed-recommended",
            "scope": "session",
            "canary": {
                "enabled": True,
                "selected": True,
                "cohort": "canary_applied",
                "fraction": 0.5,
                "pattern_hashes": ["sha256:" + "a" * 64],
            },
        }

        for index in range(2):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=10,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.01,
                cost_baseline_usd=0.02,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"category": "summary", "workflow_phase": "summary", "has_tools": False}),
                cache_json=stable_json({
                    "status": "miss",
                    "reason": "exact-miss",
                    "policy_source": "managed-recommended",
                    "pattern_rule": rule,
                    "cache_replay_canary": {
                        "schema": "agentflow.cache_replay_canary_decision.v1",
                        "rule_id": rule["rule_id"],
                        "candidate_id": rule["candidate_id"],
                        "policy_source": "managed-recommended",
                        "scope": "session",
                        "canary": rule["canary"],
                        "canary_cohort": "canary_applied",
                        "status": "applied",
                        "reason": "no-dependency-required",
                    },
                }),
                error=None,
                request_json=stable_json({
                    "messages": [{"content": "private recovery prompt /tmp/recovery-secret.py cache-key-secret"}],
                    "request_id": "req-recovery-secret",
                }),
                response_json=stable_json({"content": [{"text": "private recovery response"}]}),
                session_id=f"raw-recovery-session-secret-{index}",
                category="summary",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_streaming_cache_hit_recovery(server.store, limit=10, scan_limit=10))

        self.assertEqual(result["schema"], "agentflow.streaming_cache_hit_recovery.v1")
        self.assertTrue(result["read_only"])
        self.assertEqual(result["summary"]["recovery_verdict"], "store-missing")
        self.assertEqual(result["summary"]["eligible_calls"], 2)
        self.assertEqual(result["summary"]["replay_attempts"], 2)
        self.assertEqual(result["summary"]["successful_hits"], 0)
        self.assertEqual(result["cohorts"][0]["recovery_verdict"], "store-missing")
        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "private recovery prompt",
            "private recovery response",
            "/tmp/recovery-secret.py",
            "cache-key-secret",
            "req-recovery-secret",
            "raw-recovery-session-secret",
            "sha256:" + "a" * 64,
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(result["privacy"]["aggregate_only"])
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])

    def test_streaming_cache_hit_recovery_reports_recovered_hits(self):
        rule = {
            "rule_id": "streaming-cache-rule-healthy",
            "candidate_id": "candidate-healthy",
            "policy_source": "managed-recommended",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied", "fraction": 1.0},
        }

        def log_cache_row(status: str, *, cache_hit: int, cost: float, baseline: float) -> None:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=cache_hit,
                status_code=200,
                latency_ms=5,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"category": "summary", "workflow_phase": "summary", "has_tools": False}),
                cache_json=stable_json({
                    "status": status,
                    "reason": "exact-match" if status == "hit" else "exact-miss",
                    "hit_type": "exact" if status == "hit" else None,
                    "policy_source": "managed-recommended",
                    "pattern_rule": rule,
                    "cache_replay_canary": {
                        "schema": "agentflow.cache_replay_canary_decision.v1",
                        "rule_id": rule["rule_id"],
                        "candidate_id": rule["candidate_id"],
                        "policy_source": "managed-recommended",
                        "scope": "session",
                        "canary": rule["canary"],
                        "canary_cohort": "canary_applied",
                        "status": "applied",
                        "reason": "no-dependency-required",
                    },
                }),
                error=None,
                request_json=None,
                response_json=None,
                session_id="raw-healthy-session-secret",
                category="summary",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        log_cache_row("miss", cache_hit=0, cost=0.01, baseline=0.02)
        log_cache_row("hit", cache_hit=1, cost=0.001, baseline=0.02)

        result = asyncio.run(stats_views.stats_streaming_cache_hit_recovery(server.store, limit=10, scan_limit=10))

        self.assertEqual(result["summary"]["recovery_verdict"], "hits-recovered")
        self.assertEqual(result["summary"]["successful_hits"], 1)
        self.assertEqual(result["summary"]["replay_attempts"], 2)
        self.assertEqual(result["cohorts"][0]["hit_recovery_rate"], 0.5)

    def test_cache_replayability_report_surfaces_file_dependency_audit_reasons_without_paths(self):
        def log_tool_candidate(name, audit):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.01,
                cost_baseline_usd=0.01,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"text_chars": 8000, "category": "tool-result", "has_tools": True}),
                cache_json=stable_json({
                    "status": "miss",
                    "reason": audit["invalidation_reason"] or name,
                    "policy_source": "local-default",
                    "tool_cache_enabled": True,
                    "file_dependency_audit": audit,
                    "file_dependency_count": audit["snapshot_count"],
                    "file_dependency_evidence_available": audit["file_dependency_evidence_available"],
                    "safe_invalidation_evidence": audit["safe_invalidation_evidence"],
                }),
                error=None,
                request_json=stable_json({"messages": [{"content": "tool payload secret /tmp/private.py"}]}),
                response_json=None,
                session_id=f"session-{name}",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        base_audit = {
            "schema": "agentflow.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "cwd-relative",
            "root_path_included": False,
            "snapshot_count": 2,
            "snapshot_count_bucket": "2_5",
            "candidate_path_count_bucket": "2_5",
            "max_paths": 128,
            "cap_exceeded": False,
            "present_path_count": 2,
            "missing_path_count": 0,
            "changed_path_count": 0,
            "deleted_path_count": 0,
            "created_path_count": 0,
            "invalidation_reason": None,
            "safe_invalidation_evidence": True,
            "file_dependency_evidence_available": True,
            "paths_included": False,
        }
        log_tool_candidate("changed", {**base_audit, "changed_path_count": 1, "invalidation_reason": "dependency-changed", "safe_invalidation_evidence": False, "file_dependency_evidence_available": False})
        log_tool_candidate("deleted", {**base_audit, "deleted_path_count": 1, "invalidation_reason": "dependency-deleted", "safe_invalidation_evidence": False, "file_dependency_evidence_available": False})
        log_tool_candidate("cap", {**base_audit, "cap_exceeded": True, "invalidation_reason": "dependency-cap-exceeded", "safe_invalidation_evidence": False, "file_dependency_evidence_available": False})

        result = asyncio.run(stats_views.stats_cache_replayability(server.store, limit=10))
        rendered = json.dumps(result, sort_keys=True)
        blockers = {item["blocker"] for item in result["blocker_breakdown"]}

        self.assertIn("dependency-changed", blockers)
        self.assertIn("dependency-deleted", blockers)
        self.assertIn("dependency-cap-exceeded", blockers)
        for row in result["groups"]:
            if row["category"] == "tool-result":
                self.assertFalse(row["file_dependency_audit"]["paths_included"])
                self.assertFalse(row["file_dependency_audit"]["root_path_included"])
        self.assertNotIn("tool payload secret", rendered)
        self.assertNotIn("/tmp/private.py", rendered)
        self.assertFalse(result["privacy"]["file_paths_included"])

    def test_cache_replay_dry_run_projects_policy_hits_and_blockers_without_cache_mutation(self):
        hashes = {
            name: "sha256:" + char * 64
            for name, char in (
                ("stream", "1"),
                ("exact", "2"),
                ("tool", "3"),
                ("stale", "4"),
                ("surface", "5"),
                ("holdout", "6"),
            )
        }

        def log_candidate(
            name,
            *,
            stream,
            has_tools,
            category,
            cost,
            cache_json=None,
            crunch_json=None,
            provider="anthropic",
            path="/v1/messages",
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path=path,
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1 if stream else 0,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json(crunch_json or {"changed": False}),
                routing_json=stable_json({
                    "text_chars": 4200,
                    "category": category,
                    "workflow_phase": category,
                    "has_tools": has_tools,
                    "managed_pattern_features": {
                        "pattern_hashes": [hashes[name]],
                        "source_surface": "openai_responses" if provider == "openai" else "anthropic_messages",
                        "app_family": "claude_code",
                        "category": category,
                        "workflow_phase": category,
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "lt_1k_tokens",
                        "requested_model": "claude-sonnet-4-6",
                        "candidate_target_model": "claude-sonnet-4-6",
                        "raw_pattern_strings_included": False,
                    },
                }),
                cache_json=stable_json(cache_json or {
                    "status": "miss",
                    "reason": "exact-miss",
                    "policy_source": "local-default",
                    "replayability_level": "local-exact-response",
                }),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw dry run prompt must not leak"}]}),
                response_json=None,
                session_id=f"session-{name}-secret",
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider=provider,
            )

        streaming_cache = {
            "status": "skipped",
            "reason": "streaming",
            "policy_source": "local-default",
            "replayability_level": "local-exact-response",
        }
        log_candidate("stream", stream=True, has_tools=False, category="chat", cost=0.01, cache_json=streaming_cache)
        log_candidate("stream", stream=True, has_tools=False, category="chat", cost=0.03, cache_json=streaming_cache)
        log_candidate("exact", stream=False, has_tools=False, category="chat", cost=0.02)
        log_candidate("exact", stream=False, has_tools=False, category="chat", cost=0.04)
        log_candidate(
            "tool",
            stream=False,
            has_tools=True,
            category="tool-result",
            cost=0.05,
            cache_json={
                "status": "skipped",
                "reason": "tools-disabled",
                "policy_source": "local-default",
                "replayability_level": "local-exact-response",
            },
        )
        log_candidate(
            "stale",
            stream=False,
            has_tools=False,
            category="chat",
            cost=0.06,
            crunch_json={
                "changed": False,
                "pattern_modules": {
                    "server_features": {
                        "features": [{
                            "family": "cacheability",
                            "features": {
                                "cacheability_bucket": "low",
                                "time_sensitive_hint": True,
                                "user_specific_hint": True,
                            },
                        }],
                    },
                },
            },
        )
        log_candidate(
            "surface",
            stream=False,
            has_tools=False,
            category="chat",
            cost=0.07,
            provider="openai",
            path="/v1/responses",
        )
        log_candidate("holdout", stream=False, has_tools=False, category="chat", cost=0.08)
        server.store.set_cache("existing-cache-key", "claude-sonnet-4-6", 10, {"content": "cached"})

        proposed = {
            "cache": {
                "pattern_rules": [
                    {
                        "id": "dry-stream-rule",
                        "candidate_id": "candidate-stream",
                        "conditions": {
                            "pattern_hashes": [hashes["stream"]],
                            "source_surface": "anthropic_messages",
                            "category": "chat",
                            "has_tools": False,
                            "stream": True,
                        },
                        "action": {"type": "exact_cache_pattern", "streaming": True},
                    },
                    {
                        "id": "dry-exact-rule",
                        "candidate_id": "candidate-exact",
                        "conditions": {
                            "pattern_hashes": [hashes["exact"]],
                            "source_surface": "anthropic_messages",
                            "category": "chat",
                            "has_tools": False,
                            "stream": False,
                        },
                        "action": {"type": "exact_cache_pattern"},
                    },
                    {
                        "id": "dry-tool-rule",
                        "candidate_id": "candidate-tool",
                        "conditions": {
                            "pattern_hashes": [hashes["tool"]],
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "has_tools": True,
                            "stream": False,
                        },
                        "action": {
                            "type": "exact_cache_pattern",
                            "allow_tool_calls": True,
                            "safe_invalidation": True,
                        },
                    },
                    {
                        "id": "dry-stale-rule",
                        "candidate_id": "candidate-stale",
                        "conditions": {
                            "pattern_hashes": [hashes["stale"]],
                            "source_surface": "anthropic_messages",
                            "category": "chat",
                            "has_tools": False,
                            "stream": False,
                        },
                        "action": {"type": "exact_cache_pattern"},
                    },
                    {
                        "id": "dry-surface-rule",
                        "candidate_id": "candidate-surface",
                        "conditions": {
                            "pattern_hashes": [hashes["surface"]],
                            "source_surface": "anthropic_messages",
                            "category": "chat",
                            "has_tools": False,
                            "stream": False,
                        },
                        "action": {"type": "exact_cache_pattern"},
                    },
                    {
                        "id": "dry-holdout-rule",
                        "candidate_id": "candidate-holdout",
                        "conditions": {
                            "pattern_hashes": [hashes["holdout"]],
                            "source_surface": "anthropic_messages",
                            "category": "chat",
                            "has_tools": False,
                            "stream": False,
                        },
                        "rollout": {
                            "schema": "agentflow.pattern_policy_rollout.v1",
                            "canary_enabled": True,
                            "canary_fraction": 0.0,
                            "canary_salt": "dry-run-test",
                        },
                        "action": {"type": "exact_cache_pattern"},
                    },
                ],
            }
        }

        result = asyncio.run(stats_views.stats_cache_replay_dry_run(server.store, proposed, limit=20))

        self.assertEqual(result["schema"], "agentflow.cache_replay_dry_run.v1")
        self.assertFalse(result["summary"]["cache_table_mutated"])
        self.assertEqual(result["summary"]["cache_rows_before"], 1)
        self.assertEqual(result["summary"]["cache_rows_after"], 1)
        self.assertEqual(result["summary"]["provider_calls_made"], 0)
        self.assertEqual(result["summary"]["cache_entries_written"], 0)
        self.assertEqual(result["summary"]["candidate_rows"], 4)
        self.assertEqual(result["summary"]["projected_exact_hits"], 1)
        self.assertEqual(result["summary"]["projected_streaming_hits"], 1)
        self.assertEqual(result["summary"]["holdout_rows"], 1)
        self.assertEqual(result["summary"]["invalidation_required_rows"], 1)
        self.assertEqual(result["summary"]["unsupported_source_surface_rows"], 1)
        self.assertEqual(result["summary"]["stale_risk_blocked_rows"], 1)
        self.assertAlmostEqual(result["summary"]["estimated_saved_cost_usd"], 0.05)
        statuses = {row["status"] for row in result["rows"]}
        self.assertIn("projected-streaming-candidate", statuses)
        self.assertIn("projected-exact-candidate", statuses)
        self.assertIn("invalidation-required", statuses)
        self.assertIn("unsupported-source-surface", statuses)
        stale = next(row for row in result["rows"] if row["candidate_id"] == "candidate-stale")
        self.assertEqual(stale["reason"], "stale-risk-blockers")
        self.assertIn("current-state", stale["stale_risk_blockers"])
        tool = next(row for row in result["rows"] if row["candidate_id"] == "candidate-tool")
        self.assertEqual(tool["reason"], "file-dependency-missing")
        self.assertTrue(tool["requires_file_dependency_evidence"])
        self.assertFalse(tool["file_dependency_evidence_available"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw dry run prompt must not leak", rendered)
        self.assertNotIn("session-stream-secret", rendered)
        for pattern_hash in hashes.values():
            self.assertNotIn(pattern_hash, rendered)
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["pattern_hashes_included"])

    def test_cache_replay_dry_run_reports_dependency_freshness_for_tool_cohorts(self):
        hashes = {
            "fresh": "sha256:" + "a" * 64,
            "stale": "sha256:" + "b" * 64,
            "unknown": "sha256:" + "c" * 64,
        }
        raw_path = "/tmp/private/cache-replay-secret.py"
        raw_stat = 123456789

        def audit(*, reason=None, safe=False):
            return {
                "schema": "agentflow.cache_file_dependency_audit.v1",
                "file_watch_enabled": True,
                "snapshot_root_policy": "stored-local-paths",
                "root_path": "/tmp/private",
                "root_path_included": False,
                "paths": [raw_path],
                "snapshot_count": 1 if safe else 0,
                "snapshot_count_bucket": "1" if safe else "0",
                "candidate_path_count_bucket": "1" if safe else "0",
                "raw_candidate_path_count_bucket": "1" if safe else "0",
                "distinct_candidate_path_count_bucket": "1" if safe else "0",
                "max_paths": 16,
                "cap_exceeded": False,
                "cap_trimmed": False,
                "dependency_capture_reason": "complete",
                "present_path_count": 1 if safe else 0,
                "missing_path_count": 0 if safe else 1,
                "changed_path_count": 1 if reason == "dependency-changed" else 0,
                "deleted_path_count": 0,
                "created_path_count": 0,
                "invalidation_reason": reason,
                "safe_invalidation_evidence": safe,
                "file_dependency_evidence_available": safe,
                "mtime_ns": raw_stat,
                "paths_included": False,
            }

        def log_tool_candidate(name, *, dependency_audit, cost, created_at):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "text_chars": 4200,
                    "category": "tool-result",
                    "workflow_phase": "tool-result",
                    "has_tools": True,
                    "managed_pattern_features": {
                        "pattern_hashes": [hashes[name]],
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "category": "tool-result",
                        "workflow_phase": "tool-result",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "lt_1k_tokens",
                        "requested_model": "claude-sonnet-4-6",
                        "candidate_target_model": "claude-sonnet-4-6",
                        "raw_pattern_strings_included": False,
                    },
                }),
                cache_json=stable_json({
                    "status": "skipped",
                    "reason": "tools-disabled",
                    "policy_source": "local-default",
                    "replayability_level": "local-exact-response",
                    "file_dependency_audit": dependency_audit,
                    "file_dependency_evidence_available": bool(dependency_audit.get("file_dependency_evidence_available")),
                    "safe_invalidation_evidence": bool(dependency_audit.get("safe_invalidation_evidence")),
                }),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw tool payload must not leak"}]}),
                response_json=None,
                session_id=f"raw-session-{name}-must-not-leak",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        log_tool_candidate("fresh", dependency_audit=audit(safe=True), cost=0.02, created_at="2026-06-11T07:00:00+00:00")
        log_tool_candidate("fresh", dependency_audit=audit(safe=True), cost=0.04, created_at="2026-06-11T07:01:00+00:00")
        log_tool_candidate("stale", dependency_audit=audit(reason="dependency-changed", safe=False), cost=0.05, created_at="2026-06-11T07:02:00+00:00")
        log_tool_candidate("unknown", dependency_audit=audit(reason="file-dependency-missing", safe=False), cost=0.06, created_at="2026-06-11T07:03:00+00:00")

        proposed = {
            "cache": {
                "pattern_rules": [
                    {
                        "id": f"dry-{name}-rule",
                        "candidate_id": f"candidate-{name}",
                        "conditions": {
                            "pattern_hashes": [pattern_hash],
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "has_tools": True,
                            "stream": False,
                        },
                        "action": {
                            "type": "exact_cache_pattern",
                            "allow_tool_calls": True,
                            "safe_invalidation": True,
                        },
                    }
                    for name, pattern_hash in hashes.items()
                ],
            }
        }

        result = asyncio.run(stats_views.stats_cache_replay_dry_run(server.store, proposed, limit=20))

        self.assertEqual(result["summary"]["candidate_rows"], 2)
        self.assertEqual(result["summary"]["projected_exact_hits"], 1)
        self.assertEqual(result["summary"]["invalidation_required_rows"], 2)
        freshness = {row["value"]: row["count"] for row in result["dependency_freshness_breakdown"]}
        self.assertEqual(freshness["fresh"], 2)
        self.assertEqual(freshness["stale"], 1)
        self.assertEqual(freshness["unknown"], 1)
        fresh = next(row for row in result["rows"] if row["candidate_id"] == "candidate-fresh")
        self.assertEqual(fresh["status"], "projected-exact-candidate")
        self.assertEqual(fresh["reason"], "rule-match")
        self.assertEqual(fresh["dependency_freshness"]["status"], "fresh")
        self.assertTrue(fresh["dependency_freshness"]["safe_invalidation_evidence"])
        stale = next(row for row in result["rows"] if row["candidate_id"] == "candidate-stale")
        self.assertEqual(stale["status"], "invalidation-required")
        self.assertEqual(stale["reason"], "dependency-changed")
        self.assertEqual(stale["dependency_freshness"]["status"], "stale")
        unknown = next(row for row in result["rows"] if row["candidate_id"] == "candidate-unknown")
        self.assertEqual(unknown["status"], "invalidation-required")
        self.assertEqual(unknown["dependency_freshness"]["status"], "unknown")
        self.assertIn(unknown["reason"], {"file-dependency-missing", "file-dependency-evidence-absent"})

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(raw_path, rendered)
        self.assertNotIn(str(raw_stat), rendered)
        self.assertNotIn("raw tool payload must not leak", rendered)
        self.assertNotIn("raw-session-fresh-must-not-leak", rendered)
        for pattern_hash in hashes.values():
            self.assertNotIn(pattern_hash, rendered)
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["raw_file_stat_values_included"])

    def test_codex_effectiveness_counts_direct_derived_absent_and_unknown_model_state(self):
        rows = [
            (
                "direct-model",
                stable_json({
                    "status": "skipped",
                    "reason": "keep requested model",
                    "model_field": "model",
                    "applied": False,
                }),
                None,
            ),
            (
                "derived-model",
                stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                }),
                stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 2,
                    "method_counts": {"turn/start": 1, "initialize": 1},
                    "direction_counts": {"client_to_server": 2},
                    "model_field_state": "derived_present",
                    "model_field": "model",
                    "model_state": {
                        "state": "derived_present",
                        "field": "model",
                        "normalized_model": "gpt-5-codex",
                        "source_method": "initialize",
                        "confidence": "high",
                        "reason": "metadata-model-field",
                    },
                }),
            ),
            (
                "absent-model",
                stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                }),
                None,
            ),
            ("legacy-unknown", None, None),
        ]
        for index, (suffix, routing_json, event_window_json) in enumerate(rows):
            server.store.log_codex_app_event(
                id=f"start-{suffix}",
                created_at=f"2026-06-08T10:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-{suffix}",
                thread_id=f"thread-{suffix}",
                message_chars=120,
                params_chars=80,
                input_items=1,
                input_text_chars=64,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-model-state",
                routing_json=routing_json,
                crunch_json=stable_json({"status": "skipped", "reason": "no-change", "applied": False}),
                cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False}),
                event_window_json=event_window_json,
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        summary = result["summary"]
        breakdown = {row["value"]: row["count"] for row in result["model_field_breakdown"]}
        names = {row["value"]: row["count"] for row in result["model_field_names"]}
        derived_sample = next(
            sample for sample in result["recent_samples"]
            if sample["event_window"].get("model_field_state") == "derived_present"
        )

        self.assertEqual(summary["turn_start_rows"], 4)
        self.assertEqual(summary["model_field_present"], 2)
        self.assertEqual(summary["model_field_derived"], 1)
        self.assertEqual(summary["model_field_absent"], 1)
        self.assertEqual(summary["model_field_unknown"], 1)
        self.assertEqual(breakdown["present"], 1)
        self.assertEqual(breakdown["derived_present"], 1)
        self.assertEqual(breakdown["absent"], 1)
        self.assertEqual(breakdown["unknown"], 1)
        self.assertEqual(names["model"], 2)
        self.assertEqual(derived_sample["model_field"], "derived_present")
        self.assertEqual(derived_sample["event_window"]["model_state"]["normalized_model"], "gpt-5-codex")
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_codex_app_cache_hit_counts_decision_and_saved_cost(self):
        cache_meta = {
            "enabled": True,
            "status": "hit",
            "reason": "exact-match",
            "hit_type": "exact",
            "eligible": True,
            "policy_source": "local-default",
            "surface": "codex_app_turn",
            "replayability_level": "local-exact-response",
        }
        start_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-codex-cache",
            thread_id="thread-cache",
            message_chars=600,
            params_chars=500,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-cache-session",
            cache_json=stable_json(cache_meta),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method=None,
            request_id="req-codex-cache",
            thread_id="thread-cache",
            message_chars=100,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=5,
            session_id="codex-cache-session",
        )

        full = asyncio.run(stats_views.stats_full(server.store))
        activity = asyncio.run(stats_views.stats_activity(server.store))
        usage = asyncio.run(stats_views.stats_usage_by_owner(server.store))

        self.assertEqual(full["summary"]["today_codex_app_cost_est_usd"], 0.0)
        self.assertGreater(full["summary"]["today_codex_app_cache_savings_usd"], 0.0)
        self.assertAlmostEqual(
            full["executive_summary"]["savings"]["today_buckets"]["codex_app_exact_local_cache_usd"],
            full["summary"]["today_codex_app_cache_savings_usd"],
            places=6,
        )
        cache_rows = {
            (row["source_surface"], row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in full["today_cache_decision_breakdown"]
        }
        self.assertEqual(cache_rows[("codex_turn", "hit", "exact-match", "exact")], 1)

        codex = {unit["unit_id"]: unit for unit in activity["units"]}[f"codex_turn:{start_id}"]
        self.assertEqual(codex["replayability_level"], "local-exact-response")
        self.assertEqual(codex["optimization_features"]["cache"]["eligible"], True)
        self.assertEqual(codex["outcome_features"]["cost_est_usd"], 0.0)
        self.assertGreater(codex["outcome_features"]["cache_savings_usd"], 0.0)

        [bucket] = usage["buckets"]
        self.assertEqual(bucket["local_cache_hits"], 1)
        self.assertGreater(bucket["codex_exact_cache_savings_usd"], 0.0)
        self.assertAlmostEqual(bucket["spend_usd"], 0.0, places=6)
        self.assertGreater(bucket["captured_savings_usd"], 0.0)
        json.dumps(full)
        json.dumps(activity)
        json.dumps(usage)

    def test_error_breakdown_groups_sanitized_error_families(self):
        legacy_id = str(uuid.uuid4())
        server.store.log_call(
            id=legacy_id,
            created_at="2020-01-01T00:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=400,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=stable_json({
                "error": {
                    "message": "This model does not support the effort parameter.",
                    "type": "invalid_request_error",
                },
            }),
            request_json=None,
            response_json=None,
            session_id="session-error-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.conn.execute("update calls set provider = null where id = ?", (legacy_id,))
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-codex",
            stream=0,
            cache_hit=0,
            status_code=401,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=stable_json({
                "error": {
                    "code": "invalid_api_key",
                    "message": "Incorrect API key provided: intentionally_invalid.",
                },
            }),
            request_json=None,
            response_json=None,
            session_id="session-error-today",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="openai",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        all_time = {
            (row["provider"], row["status_code"], row["tier"], row["error_type"]): row
            for row in result["error_breakdown"]
        }
        today = {
            (row["provider"], row["status_code"], row["error_type"]): row
            for row in result["today_error_breakdown"]
        }

        legacy = all_time[("anthropic", 400, "haiku", "model_incompatible_param")]
        self.assertEqual(legacy["count"], 1)
        self.assertEqual(legacy["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(legacy["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(legacy["error_sample"], "This model does not support the effort parameter.")
        self.assertIn(("openai", 401, "auth_error"), today)
        self.assertNotIn(("anthropic", 400, "model_incompatible_param"), today)
        json.dumps(result["error_breakdown"])
        json.dumps(result["today_error_breakdown"])

    def test_routing_experiment_stats_produce_confidence_scores(self):
        for similarity, passed in ((0.9, 1), (0.7, 0)):
            server.store.log_routing_experiment(
                id=str(uuid.uuid4()),
                call_id=str(uuid.uuid4()),
                created_at=utc_now(),
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                primary_model="claude-haiku-4-5-20251001",
                shadow_model="claude-sonnet-4-6",
                category="tool-result",
                routing_reason="tool-result processing turn routed to Haiku",
                input_tokens_est=100,
                primary_status_code=200,
                shadow_status_code=200,
                primary_latency_ms=50,
                shadow_latency_ms=75,
                primary_output_chars=12,
                shadow_output_chars=14,
                primary_output_sha256="primary",
                shadow_output_sha256="shadow",
                output_similarity=similarity,
                passed_threshold=passed,
                primary_cost_est_usd=0.001,
                shadow_cost_est_usd=0.003,
                error=None,
                routing_json=stable_json({"reason": "tool-result processing turn routed to Haiku"}),
                experiment_json=stable_json({"sampled": True}),
                primary_response_json=None,
                shadow_response_json=None,
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        [row] = result["routing_experiment_summary"]

        self.assertEqual(result["summary"]["routing_experiment_samples"], 2)
        self.assertEqual(result["summary"]["routing_experiment_compared_samples"], 2)
        self.assertAlmostEqual(result["summary"]["routing_experiment_avg_similarity"], 0.8, places=6)
        self.assertEqual(result["summary"]["routing_experiment_feedback_status_counts"], {"not-exported": 2})
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["compared_samples"], 2)
        self.assertAlmostEqual(row["avg_similarity"], 0.8, places=6)
        self.assertAlmostEqual(row["pass_rate"], 0.5, places=6)
        self.assertAlmostEqual(row["confidence_score"], 0.08, places=6)
        self.assertEqual(row["min_samples_for_confidence"], 20)
        json.dumps(result["routing_experiment_summary"])

    def _log_shadow_routing_promotion_sample(
        self,
        *,
        candidate: str,
        idx: int,
        created_at: str | None = None,
        mode: str = "shadow_candidate_pass_through",
        passed: bool = True,
        shadow_status_code: int | None = 200,
        primary_status_code: int | None = 200,
        fallback: bool = False,
        error: str | None = None,
        routing_reason: str = "sampled-shadow-candidate-pass-through",
        category: str | None = None,
        workflow_phase: str = "summary",
        experiment_extra: dict | None = None,
        routing_extra: dict | None = None,
    ) -> None:
        similarity = 0.94 if passed else 0.7
        experiment_json = {
            "mode": mode,
            "counterfactual": mode == "shadow_candidate_pass_through",
            "shadow_only": mode == "shadow_candidate_pass_through",
            "workflow_phase": workflow_phase,
            "raw_prompt": "raw shadow readiness prompt",
            "response_body": "raw shadow readiness response",
            "raw_response": "raw shadow failure response",
            "provider_body": {"messages": [{"content": "raw shadow failure prompt"}]},
            "request_id": "req-shadow-secret",
            "session_id": "session-shadow-secret",
            "tenant_id": "tenant-shadow-secret",
            "account_id": "account-shadow-secret",
            "file_path": "/tmp/shadow-secret.py",
            "cache_key": "cache-shadow-secret",
            "tool_payload": {"secret": "tool-shadow-secret"},
            "authorization": "Bearer authorization-shadow-secret",
            "api_key": "sk-shadow-secret",
        }
        if experiment_extra:
            experiment_json.update(experiment_extra)
        routing_json = {"workflow_phase": workflow_phase}
        if routing_extra:
            routing_json.update(routing_extra)
        if fallback:
            routing_json["fallback_reason"] = "rate_limited"
        server.store.log_routing_experiment(
            id=f"shadow-readiness-{candidate}-{idx}",
            call_id=f"shadow-call-secret-{candidate}-{idx}",
            created_at=created_at or utc_now(),
            provider="openai",
            source_surface="codex_turn",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            primary_model="gpt-5-codex",
            shadow_model="gpt-5-mini",
            category=category or f"codex-{candidate}",
            routing_reason=routing_reason,
            input_tokens_est=100,
            primary_status_code=primary_status_code,
            shadow_status_code=shadow_status_code,
            primary_latency_ms=120,
            shadow_latency_ms=80,
            primary_output_chars=20,
            shadow_output_chars=18,
            primary_output_sha256=f"primary-{candidate}-{idx}",
            shadow_output_sha256=f"shadow-{candidate}-{idx}",
            output_similarity=similarity if shadow_status_code and shadow_status_code < 400 else None,
            passed_threshold=1 if passed else 0,
            primary_cost_est_usd=0.003,
            shadow_cost_est_usd=0.001,
            routing_json=stable_json(routing_json),
            experiment_json=stable_json(experiment_json),
            error=error,
        )

    def _patch_shadow_routing_promotion_thresholds(self):
        return patch.multiple(
            routing_experiments,
            ROUTING_EXPERIMENT_ENABLED=True,
            ROUTING_EXPERIMENT_MIN_SAMPLES=3,
            ROUTING_EXPERIMENT_SAMPLE_RATE=0.25,
            ROUTING_EXPERIMENT_DAILY_BUDGET_USD=10.0,
            ROUTING_EXPERIMENT_POLICY={
                "profile_id": "test-shadow-readiness",
                "mode": "shadow_candidate_pass_through",
                "sample_rate": 0.25,
                "daily_budget_usd": 10.0,
                "min_samples_for_confidence": 3,
                "providers": ["openai"],
                "source_surfaces": ["codex_turn"],
                "model_pairs": [{"requested_model": "gpt-5-codex", "routed_model": "gpt-5-mini"}],
                "categories": [],
                "workflow_phases": [],
            },
        )

    def test_shadow_routing_promotion_readiness_endpoint_and_dashboard_are_metadata_only(self):
        stale = "2025-01-01T00:00:00+00:00"
        with self._patch_shadow_routing_promotion_thresholds():
            for idx in range(3):
                self._log_shadow_routing_promotion_sample(candidate="promote", idx=idx)
                self._log_shadow_routing_promotion_sample(candidate="hold", idx=idx, created_at=stale)
                self._log_shadow_routing_promotion_sample(candidate="reject", idx=idx, passed=False)
                self._log_shadow_routing_promotion_sample(candidate="applied", idx=idx, mode="applied_routed_down")
            self._log_shadow_routing_promotion_sample(candidate="needs", idx=0)

            result = asyncio.run(stats_views.stats_shadow_routing_promotion_readiness(server.store, limit=20))
            app = create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.example.invalid",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            endpoint = client.get("/agentflow/stats/shadow-routing-promotion-readiness?limit=20")
            html = client.get("/agentflow/dashboard")

        self.assertEqual(result["schema"], "agentflow.shadow_routing_promotion_readiness.v1")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["wrote_local_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        verdicts = {row["promotion_verdict"] for row in result["candidates"]}
        self.assertIn("promote", verdicts)
        self.assertIn("hold", verdicts)
        self.assertIn("needs_more_samples", verdicts)
        self.assertIn("reject", verdicts)
        self.assertGreater(result["summary"]["applied_canary_count"], 0)
        self.assertGreater(result["summary"]["shadow_only_count"], 0)
        applied = next(row for row in result["candidates"] if row["sample_mode"] == "applied_routed_down")
        holdout = next(row for row in result["candidates"] if row["sample_mode"] == "shadow_candidate_pass_through")
        self.assertTrue(applied["canary"]["canary_evidence"])
        self.assertGreater(applied["canary"]["applied_count"], 0)
        self.assertTrue(holdout["canary"]["shadow_only"])
        self.assertGreater(holdout["canary"]["holdout_count"], 0)
        self.assertIn("readiness_state", applied)
        self.assertIn("fallback_or_retry_count", applied)
        self.assertIn("avg_latency_delta_ms", applied)
        self.assertEqual(endpoint.status_code, 200)
        endpoint_verdicts = {row["promotion_verdict"] for row in endpoint.json()["candidates"]}
        self.assertEqual(endpoint_verdicts, verdicts)
        self.assertEqual(html.status_code, 200)
        self.assertIn("/agentflow/stats/shadow-routing-promotion-readiness", html.text)
        self.assertIn("Shadow-routing promotion readiness", html.text)
        self.assertIn("shadow-routing-promotion-candidates-tbody", html.text)
        rendered = json.dumps(endpoint.json(), sort_keys=True) + html.text
        self.assertIn("metadata only", html.text)
        self._assert_shadow_promotion_forbidden_absent(rendered)

    def test_shadow_routing_promotion_failure_fixtures_are_bounded_and_metadata_only(self):
        stale = "2025-01-01T00:00:00+00:00"
        with self._patch_shadow_routing_promotion_thresholds():
            for idx in range(3):
                self._log_shadow_routing_promotion_sample(candidate="low-similarity", idx=idx, passed=False)
                self._log_shadow_routing_promotion_sample(
                    candidate="shadow-error",
                    idx=idx,
                    shadow_status_code=500,
                    error="raw-shadow-error-body should not leak",
                )
                self._log_shadow_routing_promotion_sample(
                    candidate="primary-error",
                    idx=idx,
                    primary_status_code=500,
                )
                self._log_shadow_routing_promotion_sample(candidate="fallback", idx=idx, fallback=True)
                self._log_shadow_routing_promotion_sample(candidate="stale", idx=idx, created_at=stale)
                self._log_shadow_routing_promotion_sample(
                    candidate="raw-labels",
                    idx=idx,
                    routing_reason="routing reason raw secret /tmp/shadow-secret.py",
                    category="category raw secret",
                    workflow_phase="workflow raw secret",
                    experiment_extra={
                        "mode": "mode raw secret",
                        "managed_feedback": {"status": "queued raw secret", "workflow_phase": "managed workflow raw secret"},
                    },
                    routing_extra={
                        "routing_experiment": {"status": "status raw secret", "reason": "reason raw secret"},
                        "safety_stop": {
                            "tripped": True,
                            "reason": "safety stop raw secret",
                            "reason_codes": ["safety-stop-observed raw secret"],
                        },
                    },
                )

            result = asyncio.run(stats_views.stats_shadow_routing_promotion_readiness(server.store, limit=20))
            app = create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.example.invalid",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            endpoint = client.get("/agentflow/stats/shadow-routing-promotion-readiness?limit=20")
            html = client.get("/agentflow/dashboard")

        self.assertEqual(endpoint.status_code, 200)
        self.assertEqual(html.status_code, 200)
        candidates = {row["category"]: row for row in result["candidates"]}
        self.assertEqual(candidates["codex-low-similarity"]["promotion_verdict"], "reject")
        self.assertIn("below-similarity-pass-rate", candidates["codex-low-similarity"]["reason_codes"])
        self.assertEqual(candidates["codex-shadow-error"]["promotion_verdict"], "reject")
        self.assertIn("shadow-error-rate-high", candidates["codex-shadow-error"]["reason_codes"])
        self.assertNotEqual(candidates["codex-primary-error"]["promotion_verdict"], "promote")
        self.assertIn("primary-error-rate-high", candidates["codex-primary-error"]["reason_codes"])
        self.assertEqual(candidates["codex-fallback"]["promotion_verdict"], "hold")
        self.assertIn("fallback-or-retry-observed", candidates["codex-fallback"]["reason_codes"])
        self.assertEqual(candidates["codex-stale"]["promotion_verdict"], "hold")
        self.assertIn("stale-evidence", candidates["codex-stale"]["reason_codes"])
        redacted = candidates["redacted-metadata-label"]
        self.assertEqual(redacted["workflow_phase"], "redacted-metadata-label")
        self.assertEqual(redacted["sample_mode"], "redacted-metadata-label")
        self.assertEqual(redacted["routing_reasons"][0]["reason"], "redacted-metadata-label")
        rendered = json.dumps(result, sort_keys=True) + json.dumps(endpoint.json(), sort_keys=True) + html.text
        self._assert_shadow_promotion_forbidden_absent(rendered)
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])

    def test_sessions_include_thinking_token_breakdown(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.02,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-thinking",
            category="tool-heavy",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=1_000,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]

        self.assertEqual(session["session_id"], "session-thinking")
        self.assertEqual(session["thinking_tokens"], 1_000)
        self.assertAlmostEqual(session["thinking_cost_usd"], 0.015, places=6)
        json.dumps(result)

    def test_sessions_include_prompt_cache_warmup_breakdown(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.02,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-warmup",
            category="tool-heavy",
            cache_creation_input_tokens=1_000,
            cache_read_input_tokens=10_000,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]

        self.assertEqual(session["session_id"], "session-cache-warmup")
        self.assertEqual(session["cache_creation_tokens"], 1_000)
        self.assertEqual(session["cache_read_tokens"], 10_000)
        self.assertEqual(session["cache_write_read_token_ratio"], 0.1)
        self.assertAlmostEqual(session["cache_creation_cost_usd"], 0.00375, places=6)
        self.assertAlmostEqual(session["cache_read_savings_usd"], 0.027, places=6)
        self.assertEqual(session["cache_warmup_payback_ratio"], 0.139)
        json.dumps(result)

    def test_sessions_include_codex_app_turns_without_raw_payloads(self):
        raw_prompt_text = "secret raw prompt must not appear"

        def log_codex_turn(
            request_id: str,
            *,
            input_text_chars: int,
            result_chars: int | None,
            routing: dict | None = None,
            crunch: dict | None = None,
            cache: dict | None = None,
            error_code: int | None = None,
            latency_ms: int = 100,
        ) -> None:
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id="thread-codex-sessions",
                message_chars=input_text_chars + 50,
                params_chars=input_text_chars + 20,
                input_items=2,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=None,
                routing_json=stable_json(routing) if routing is not None else None,
                crunch_json=stable_json(crunch) if crunch is not None else None,
                cache_json=stable_json(cache) if cache is not None else None,
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id="thread-codex-sessions",
                message_chars=(result_chars or 0) + 20,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=error_code,
                error_message="metadata-only error" if error_code is not None else None,
                latency_ms=latency_ms,
                session_id=None,
            )

        log_codex_turn(
            "codex-s1",
            input_text_chars=10_000,
            result_chars=400,
            routing={
                "applied": True,
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.1-codex",
                "policy_source": "local-manual",
            },
            crunch={
                "changed": True,
                "applied": True,
                "saved_chars": 1_200,
                "policy_source": "local-default",
            },
            cache={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "local-default",
            },
        )
        log_codex_turn(
            "codex-s2",
            input_text_chars=10_200,
            result_chars=200,
            cache={
                "status": "hit",
                "reason": "exact-match",
                "hit_type": "exact",
                "policy_source": "local-default",
            },
        )
        log_codex_turn(
            "codex-s3",
            input_text_chars=14_000,
            result_chars=None,
            error_code=-32000,
        )

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]
        encoded = json.dumps(result)

        self.assertTrue(session["session_id"].startswith("codex-workflow:"))
        self.assertEqual(session["session_key_basis"], "workflow_thread_id")
        self.assertEqual(
            session["codex_workflow_grouping"]["original_key_basis_counts"],
            {"thread_id": 3},
        )
        self.assertEqual(session["codex_workflow_grouping"]["original_key_count"], 1)
        self.assertFalse(session["codex_workflow_grouping"]["raw_keys_included"])
        self.assertEqual(session["source_surface"], "codex_turn")
        self.assertEqual(session["app_family"], "codex")
        self.assertEqual(session["calls"], 3)
        self.assertEqual(session["provider_calls"], 0)
        self.assertEqual(session["codex_turns"], 3)
        self.assertEqual(session["codex_input_text_chars"], 34_200)
        self.assertEqual(session["codex_input_tokens_est"], 8_550)
        self.assertEqual(session["codex_output_tokens_est"], 150)
        self.assertEqual(session["codex_total_tokens_est"], 8_700)
        self.assertGreater(session["codex_cost_est_usd"], 0.0)
        self.assertGreater(session["codex_baseline_cost_est_usd"], session["codex_cost_est_usd"])
        self.assertGreater(session["codex_exact_cache_savings_usd"], 0.0)
        self.assertEqual(session["codex_routed_turns"], 1)
        self.assertEqual(session["codex_crunched_turns"], 1)
        self.assertEqual(session["codex_cache_hits"], 1)
        self.assertEqual(session["codex_optimized_turns"], 2)
        self.assertEqual(session["codex_errors"], 1)
        self.assertEqual(session["codex_method_counts"], [{"method": "turn/start", "turns": 3}])
        self.assertEqual(plateau["session_id"], session["session_id"])
        self.assertEqual(plateau["session_key_basis"], "workflow_thread_id")
        self.assertEqual(plateau["source_surface"], "codex_turn")
        self.assertEqual(plateau["calls"], 3)
        self.assertEqual(plateau["plateau_pairs"], 1)
        self.assertEqual(plateau["median_text_chars"], 10_200)
        self.assertEqual(plateau["p90_text_chars"], 14_000)
        self.assertEqual(plateau["crunch_saved_chars"], 1_200)
        self.assertGreater(plateau["cache_read_savings_usd"], 0.0)
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("raw_prompt", encoded)
        self.assertNotIn("thread-codex-sessions", encoded)

    def test_sessions_group_fragmented_codex_turns_into_workflow_window(self):
        raw_prompt_text = "secret raw codex prompt must not appear"
        created_base = utc_now()
        text_sizes = [10_000, 10_100, 10_050, 10_080, 10_060]
        phases = ["planning", "planning", "tool_execution", "summary", "summary"]
        for idx, text_chars in enumerate(text_sizes):
            request_id = f"request-fragment-{idx}"
            session_id = f"codex-fragment-session-{idx}"
            server.store.log_codex_app_event(
                id=f"fragment-start-{idx}",
                created_at=created_base,
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=None,
                message_chars=text_chars + 25,
                params_chars=text_chars + 10,
                input_items=2,
                input_text_chars=text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=session_id,
                routing_json=stable_json({
                    "status": "skipped",
                    "applied": False,
                    "workflow_phase": phases[idx],
                    "reason": "test-metadata-only",
                }),
                crunch_json=stable_json({
                    "changed": idx == 2,
                    "saved_chars": 400 if idx == 2 else 0,
                    "workflow_phase": phases[idx],
                }),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            )
            server.store.log_codex_app_event(
                id=f"fragment-response-{idx}",
                created_at=created_base,
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=None,
                message_chars=200,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=200,
                error_code=None,
                error_message=None,
                latency_ms=100 + idx,
                session_id=session_id,
            )

        app = create_dashboard_app(
            store_obj=lambda: server.store,
            default_db=self.tmp.name,
            upstream="https://anthropic.test",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        response = TestClient(app).get("/agentflow/stats/sessions")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]
        encoded = json.dumps(result)

        self.assertTrue(session["session_id"].startswith("codex-workflow:"))
        self.assertEqual(session["session_key_basis"], "workflow_window")
        self.assertEqual(session["source_surface"], "codex_turn")
        self.assertEqual(session["app_family"], "codex")
        self.assertEqual(session["calls"], 5)
        self.assertEqual(session["codex_turns"], 5)
        self.assertEqual(session["provider_calls"], 0)
        self.assertEqual(session["codex_input_text_chars"], sum(text_sizes))
        self.assertGreater(session["codex_cost_est_usd"], 0.0)
        self.assertEqual(session["codex_crunched_turns"], 1)
        self.assertEqual(
            session["codex_workflow_grouping"]["original_key_basis_counts"],
            {"session_id": 5},
        )
        self.assertEqual(session["codex_workflow_grouping"]["original_key_count"], 5)
        self.assertFalse(session["codex_workflow_grouping"]["raw_keys_included"])
        self.assertEqual(
            {row["phase"]: row["turns"] for row in session["codex_workflow_phase_counts"]},
            {"planning": 2, "summary": 2, "tool_execution": 1},
        )
        self.assertEqual(plateau["session_id"], session["session_id"])
        self.assertEqual(plateau["session_key_basis"], "workflow_window")
        self.assertEqual(plateau["calls"], 5)
        self.assertEqual(plateau["plateau_pairs"], 4)
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("raw_prompt", encoded)
        self.assertNotIn("raw_response", encoded)
        self.assertNotIn("codex-fragment-session-", encoded)
        self.assertNotIn("request-fragment-", encoded)

    def test_recent_session_spending_summary_breaks_down_cost_drivers(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=1_000,
            output_tokens_est=100,
            actual_input_tokens=1_000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"reason": "test route"}),
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-spending",
            category="tool-result",
            cache_creation_input_tokens=500,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=300,
            provider="anthropic",
        )

        [summary] = server._recent_session_spending_summary()

        self.assertEqual(summary["session_id"], "session-spending")
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["cache_creation_tokens"], 500)
        self.assertEqual(summary["cache_read_tokens"], 2_000)
        self.assertEqual(summary["thinking_tokens"], 300)
        self.assertAlmostEqual(summary["cost_usd"], 0.001, places=6)
        self.assertAlmostEqual(summary["baseline_savings_usd"], 0.009, places=6)
        self.assertAlmostEqual(summary["routing_savings_usd"], 0.003, places=6)
        self.assertAlmostEqual(summary["prompt_cache_savings_usd"], 0.0018, places=6)
        self.assertAlmostEqual(summary["thinking_cost_usd"], 0.0015, places=6)
        json.dumps(summary)

    def test_provider_prompt_cache_accounting_uses_provider_pricing_and_stays_out_of_local_cache_savings(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="anthropic-prompt-cache",
            category="chat",
            cache_creation_input_tokens=1_000,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-codex",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=2_000,
            actual_output_tokens=10,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="openai-prompt-cache",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]
        prompt_cache = result["provider_prompt_cache_accounting"]
        by_provider = {row["provider"]: row for row in prompt_cache["by_model"]}
        savings = {
            (row["source_surface"], row["optimization_type"]): row["savings_usd"]
            for row in result["today_savings_by_source_surface"]
        }

        self.assertEqual(prompt_cache["label"], "provider prompt-cache discount/economics")
        self.assertIn("separate from AgentFlow local exact-cache", prompt_cache["boundary"])
        self.assertAlmostEqual(by_provider["anthropic"]["read_discount_usd"], 0.0054, places=8)
        self.assertAlmostEqual(by_provider["anthropic"]["actual_cached_read_cost_usd"], 0.0006, places=8)
        self.assertAlmostEqual(by_provider["anthropic"]["creation_cost_usd"], 0.00375, places=8)
        self.assertAlmostEqual(by_provider["anthropic"]["creation_premium_usd"], 0.00075, places=8)
        self.assertEqual(by_provider["anthropic"]["pricing_source"], "embedded-agentflow-defaults")
        self.assertEqual(by_provider["anthropic"]["pricing_version"], "2026-06-08")
        self.assertAlmostEqual(by_provider["openai"]["read_discount_usd"], 0.00225, places=8)
        self.assertAlmostEqual(by_provider["openai"]["actual_cached_read_cost_usd"], 0.00025, places=8)
        self.assertAlmostEqual(summary["provider_prompt_cache_discount_usd"], 0.00765, places=8)
        self.assertAlmostEqual(summary["provider_prompt_cache_net_discount_usd"], 0.0069, places=8)
        self.assertAlmostEqual(summary["cache_savings_usd"], 0.0, places=8)
        self.assertNotIn(("anthropic_messages", "cache"), savings)
        self.assertNotIn(("openai_responses", "cache"), savings)
        self.assertAlmostEqual(savings[("anthropic_messages", "provider_prompt_cache")], 0.0054, places=8)
        self.assertAlmostEqual(savings[("openai_responses", "provider_prompt_cache")], 0.00225, places=8)
        # Acceptance criteria: provider prompt-cache discount must NOT be in AgentFlow headline savings
        exec_savings = result["executive_summary"]["savings"]
        self.assertAlmostEqual(exec_savings["today_agentflow_generated_savings_usd"], 0.0, places=6,
                               msg="provider prompt-cache discount must not inflate AgentFlow headline savings")
        self.assertGreater(exec_savings["today_provider_prompt_cache_discount_usd"], 0.0,
                           msg="provider prompt-cache discount must appear in separate provider section")
        ppc = exec_savings["provider_prompt_cache_economics"]
        self.assertGreater(ppc["today_read_discount_usd"], 0.0)
        self.assertIn("provider billing efficiency", ppc["label"])
        self.assertNotIn("provider_prompt_cache_discount_usd", exec_savings["today_agentflow_generated_buckets"])
        json.dumps(result)

    def test_agentflow_generated_savings_excludes_provider_prompt_cache(self):
        # A call with routing+crunch savings AND provider prompt-cache reads.
        # agentflow_generated_savings_usd must equal only routing+crunch+cache, not prompt-cache.
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-opus-4-8",
            routed_model="claude-haiku-4-5-20251001",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=5,
            input_tokens_est=1_000,
            output_tokens_est=100,
            actual_input_tokens=1_000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 400}),
            routing_json=stable_json({"reason": "cost-route"}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="split-savings-test",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=1_000,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        result = asyncio.run(stats_views.stats_full(server.store))
        exec_savings = result["executive_summary"]["savings"]
        agentflow_total = exec_savings["today_agentflow_generated_savings_usd"]
        ppc_discount = exec_savings["today_provider_prompt_cache_discount_usd"]
        # prompt-cache discount > 0 (1000 cache read tokens for claude-sonnet at 0.3/MTok = $0.0003)
        self.assertGreater(ppc_discount, 0.0)
        # routing savings > 0 (routing from opus to haiku cuts baseline by 9x)
        self.assertGreater(exec_savings["today_agentflow_generated_buckets"]["routing_usd"], 0.0)
        # agentflow total must not include the ppc_discount
        self.assertAlmostEqual(agentflow_total, exec_savings["today_agentflow_generated_buckets"]["routing_usd"] + exec_savings["today_agentflow_generated_buckets"]["crunching_usd"] + exec_savings["today_agentflow_generated_buckets"]["exact_local_cache_usd"], places=6)
        self.assertNotIn("provider_prompt_cache_discount_usd", exec_savings["today_agentflow_generated_buckets"])
        self.assertIn("today_provider_prompt_cache_discount_usd", exec_savings)

    def test_stats_full_exposes_cache_replay_cohort_ranking_for_research_handoff(self):
        dependency_audit = {
            "schema": "agentflow.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "stored-local-paths",
            "root_path_included": False,
            "snapshot_count": 2,
            "snapshot_count_bucket": "2_5",
            "candidate_path_count_bucket": "2_5",
            "raw_candidate_path_count_bucket": "2_5",
            "distinct_candidate_path_count_bucket": "2_5",
            "dependency_capture_reason": "complete",
            "present_path_count": 2,
            "missing_path_count": 0,
            "changed_path_count": 0,
            "deleted_path_count": 0,
            "created_path_count": 0,
            "safe_invalidation_evidence": True,
            "file_dependency_evidence_available": True,
            "paths": ["/home/lutz/private/cache-source.py"],
            "paths_included": True,
        }
        for index, cost in enumerate((0.02, 0.03)):
            server.store.log_call(
                id=f"cache-replay-secret-call-{index}",
                created_at=f"2026-06-10T04:0{index}:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=4000,
                output_tokens_est=200,
                actual_input_tokens=4000,
                actual_output_tokens=200,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "has_tools": True,
                    "text_chars": 64000,
                    "managed_pattern_features": {
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "text_bucket": "32k_128k_chars",
                        "pattern_hashes": ["sha256:" + "a" * 64],
                    },
                }),
                cache_json=stable_json({
                    "status": "skipped",
                    "reason": "streaming",
                    "policy_source": "local-default",
                    "tool_cache_enabled": False,
                    "replayability_level": "local-exact-response",
                    "replay_scope": "session",
                    "replay_scope_id_available": True,
                    "cache_key": "raw-cache-key-secret",
                    "file_dependency_audit": dependency_audit,
                }),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw replay prompt must not leak"}]}),
                response_json=stable_json({"content": "raw replay response must not leak"}),
                session_id="raw-replay-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        ranking = result["cache_replay_cohort_ranking"]

        self.assertEqual(ranking["schema"], "agentflow.cache_replay_plateau_cohort_ranking.v1")
        self.assertEqual(ranking["summary"]["activation_ready_count"], 1)
        self.assertEqual(ranking["summary"]["projected_ready_hits"], 1)
        self.assertGreaterEqual(ranking["summary"]["projected_ready_saved_cost_usd"], 0.02)
        self.assertEqual(ranking["cohorts"][0]["readiness"], "activation-ready")
        self.assertEqual(ranking["cohorts"][0]["dependency_state"], "stable")
        self.assertEqual(ranking["cohorts"][0]["projected_hits"], 1)
        self.assertTrue(ranking["cohorts"][0]["recommended_canary"]["safe_invalidation"])
        rendered = json.dumps(ranking, sort_keys=True)
        for forbidden in (
            "raw replay prompt must not leak",
            "raw replay response must not leak",
            "raw-replay-session-secret",
            "raw-cache-key-secret",
            "cache-replay-secret-call-",
            "/home/lutz/private",
            "sha256:" + "a" * 64,
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(ranking["privacy"]["raw_request_bodies_included"])
        self.assertFalse(ranking["privacy"]["raw_session_ids_included"])
        self.assertFalse(ranking["privacy"]["cache_keys_included"])
        self.assertFalse(ranking["privacy"]["pattern_hashes_included"])

    def test_activity_stats_normalize_provider_calls_and_codex_turns(self):
        provider_id = str(uuid.uuid4())
        server.store.log_call(
            id=provider_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=12,
            input_tokens_est=100,
            output_tokens_est=20,
            actual_input_tokens=90,
            actual_output_tokens=18,
            cost_est_usd=0.001,
            cost_baseline_usd=0.004,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 10, "policy_source": "local-default"}),
            routing_json=stable_json({
                "reason": "tool result routed to Haiku",
                "text_chars": 360,
                "has_tools": True,
                "category": "tool-result",
                "policy_source": "local-manual",
            }),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="provider-session",
            category="tool-result",
            cache_creation_input_tokens=5,
            cache_read_input_tokens=50,
            retry_count=1,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        start_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        raw_prompt_text = "secret raw prompt must not appear"
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-1",
            thread_id="thread-1",
            message_chars=500,
            params_chars=450,
            input_items=2,
            input_text_chars=len(raw_prompt_text),
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-session",
        )
        server.store.log_codex_app_event(
            id=response_id,
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-1",
            thread_id="thread-1",
            message_chars=300,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=200,
            error_code=None,
            error_message=None,
            latency_ms=3000,
            session_id="codex-session",
        )

        result = asyncio.run(stats_views.stats_activity(server.store))
        units = {unit["unit_id"]: unit for unit in result["units"]}
        provider = units[f"provider_call:{provider_id}"]
        codex = units[f"codex_turn:{start_id}"]

        self.assertEqual(result["schema"], "agentflow.optimization_activity.v1")
        self.assertEqual(provider["source_surface"], "anthropic_messages")
        self.assertEqual(provider["granularity"], "provider_request")
        self.assertEqual(provider["app_family"], "claude_code")
        self.assertEqual(provider["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(provider["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(provider["tool_features"]["has_tools"], True)
        self.assertEqual(provider["input_features"]["text_chars"], 360)
        self.assertEqual(provider["input_features"]["input_tokens"], 90)
        self.assertEqual(provider["optimization_features"]["cache"]["status"], "skipped")
        self.assertEqual(provider["optimization_features"]["crunch"]["changed"], True)
        self.assertEqual(provider["optimization_features"]["policy_sources"], ["local-default", "local-manual"])
        self.assertEqual(provider["outcome_features"]["status_code"], 200)
        self.assertEqual(provider["outcome_features"]["cost_est_usd"], 0.001)
        self.assertEqual(provider["replayability_level"], "features_only")
        self.assertEqual(provider["local_ids"]["calls_id"], provider_id)

        self.assertEqual(codex["schema"], "agentflow.optimization_unit.v1")
        self.assertEqual(codex["source_surface"], "codex_turn")
        self.assertEqual(codex["granularity"], "agent_turn")
        self.assertEqual(codex["app_family"], "codex")
        self.assertEqual(codex["requested_model"], stats_views.CODEX_APP_MODEL)
        self.assertEqual(codex["target_model"], stats_views.CODEX_APP_MODEL)
        self.assertEqual(codex["model_basis"], "estimated")
        self.assertEqual(codex["input_features"]["category"], "codex-app-turn")
        self.assertEqual(codex["input_features"]["input_text_chars"], len(raw_prompt_text))
        self.assertEqual(codex["input_features"]["input_tokens_est"], 8)
        self.assertEqual(codex["input_features"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertEqual(codex["tool_features"]["mutation_safe"], False)
        self.assertEqual(codex["tool_features"]["mutation_safe_reason"], "codex-app-telemetry-only")
        self.assertEqual(codex["tool_features"]["tool_or_approval_hints"]["captured"], False)
        self.assertEqual(codex["risk_features"]["params_shape"]["has_params"], True)
        self.assertEqual(codex["risk_features"]["params_shape"]["input_items"], 2)
        self.assertEqual(codex["risk_features"]["raw_prompt_stored"], False)
        self.assertEqual(codex["mutation_safe"], False)
        self.assertEqual(codex["optimization_features"]["routing"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["routing"]["reason"], "codex-app-telemetry-only")
        self.assertEqual(codex["optimization_features"]["crunch"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["cache"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["policy_sources"], ["local-default"])
        self.assertEqual(codex["outcome_features"]["status"], "success")
        self.assertEqual(codex["outcome_features"]["latency_ms"], 3000)
        self.assertEqual(codex["outcome_features"]["output_tokens_est"], 50)
        self.assertEqual(codex["outcome_features"]["total_tokens_est"], 58)
        self.assertEqual(codex["outcome_features"]["pricing_basis"]["model"], "gpt-5.3-codex")
        self.assertAlmostEqual(codex["outcome_features"]["cost_est_usd"], 0.000714, places=6)
        self.assertAlmostEqual(codex["outcome_features"]["cost_baseline_usd"], 0.000714, places=6)
        self.assertAlmostEqual(codex["outcome_features"]["hard_floor_usd"], 0.000714, places=6)
        self.assertEqual(codex["outcome_features"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertEqual(codex["replayability_level"], "features_only")
        self.assertEqual(codex["local_ids"]["codex_app_response_event_id"], response_id)
        self.assertNotIn(raw_prompt_text, json.dumps(codex))
        self.assertEqual(result["summary"]["provider_request_units"], 1)
        self.assertEqual(result["summary"]["codex_turn_units"], 1)
        self.assertEqual(result["summary"]["codex_app_turn_units"], 1)
        self.assertEqual(result["summary"]["by_source_surface"]["anthropic_messages"], 1)
        self.assertEqual(result["summary"]["by_source_surface"]["codex_turn"], 1)
        json.dumps(result)

    def test_activity_stats_use_codex_app_policy_metadata_when_recorded(self):
        start_id = str(uuid.uuid4())
        routing_meta = {
            "enabled": True,
            "status": "applied",
            "applied": True,
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-haiku-4-5-20251001",
            "reason": "small non-tool Sonnet request routed to Haiku",
            "policy_source": "local-manual",
            "surface": "codex_app_turn",
        }
        crunch_meta = {
            "enabled": True,
            "status": "applied",
            "changed": True,
            "saved_chars": 400,
            "tokens_before_est": 200,
            "tokens_saved_est": 100,
            "policy_source": "local-default",
            "surface": "codex_app_turn",
        }
        cache_meta = {
            "enabled": True,
            "status": "not-applied",
            "reason": "codex-app-cache-not-implemented",
            "policy_source": "local-default",
            "surface": "codex_app_turn",
        }
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-policy",
            thread_id="thread-policy",
            message_chars=300,
            params_chars=250,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-policy",
            routing_json=stable_json(routing_meta),
            crunch_json=stable_json(crunch_meta),
            cache_json=stable_json(cache_meta),
        )

        result = asyncio.run(stats_views.stats_activity(server.store))
        codex = {unit["unit_id"]: unit for unit in result["units"]}[f"codex_turn:{start_id}"]

        self.assertEqual(codex["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(codex["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(codex["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(codex["optimization_features"]["routing"]["status"], "applied")
        self.assertEqual(codex["optimization_features"]["crunch"]["saved_chars"], 400)
        self.assertEqual(codex["optimization_features"]["cache"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["policy_sources"], ["local-default", "local-manual"])
        self.assertEqual(codex["replayability_level"], "features_only")
        json.dumps(result)

    def test_quality_signal_summary_uses_metadata_only_provider_and_codex_rows(self):
        def log_provider(status_code, *, retry_count=0, error=None, routing=None):
            call_id = str(uuid.uuid4())
            server.store.log_call(
                id=call_id,
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model=(routing or {}).get("routed_model") or "claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=100,
                input_tokens_est=10,
                output_tokens_est=2,
                actual_input_tokens=10,
                actual_output_tokens=2,
                cost_est_usd=0.001,
                cost_baseline_usd=0.001,
                crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
                routing_json=stable_json(routing or {"policy_source": "local-default"}),
                cache_json=stable_json({"status": "miss", "policy_source": "local-default"}),
                error=error,
                request_json=None,
                response_json=None,
                session_id="quality-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
            )
            return call_id

        log_provider(
            200,
            retry_count=1,
            routing={
                "applied": True,
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "policy_source": "local-default",
            },
        )
        log_provider(400)
        log_provider(429, error="temporarily limiting requests for tier sonnet")

        abandoned_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=abandoned_id,
            created_at="2000-01-01T00:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="quality-abandoned",
            thread_id="quality-thread",
            message_chars=100,
            params_chars=100,
            input_items=1,
            input_text_chars=100,
            session_id="quality-codex",
            routing_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            crunch_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
        )
        pending_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=pending_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="quality-pending",
            thread_id="quality-thread",
            message_chars=100,
            params_chars=100,
            input_items=1,
            input_text_chars=100,
            session_id="quality-codex",
            routing_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            crunch_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
        )

        activity = asyncio.run(stats_views.stats_activity(server.store, limit=20))
        quality = asyncio.run(stats_views.stats_quality_signals(server.store, limit=20))

        summary = activity["summary"]["quality_signal_summary"]
        by_status = {row["status"]: row["count"] for row in summary["by_status"]}
        by_signal = {row["signal"]: row["count"] for row in summary["by_signal"]}
        self.assertEqual(quality["schema"], "agentflow.quality_signal_report.v1")
        self.assertFalse(quality["privacy"]["raw_prompts_included"])
        self.assertEqual(quality["summary"], summary)
        self.assertGreaterEqual(by_status["success"], 1)
        self.assertGreaterEqual(by_status["failure"], 1)
        self.assertGreaterEqual(by_status["local_throttled"], 1)
        self.assertGreaterEqual(by_status["abandoned"], 1)
        self.assertGreaterEqual(by_status["pending"], 1)
        self.assertGreaterEqual(by_signal["retry-after-error"], 1)
        self.assertGreaterEqual(by_signal["local-throttled"], 1)
        self.assertGreaterEqual(by_signal["abandoned"], 1)
        self.assertNotIn("request_json", json.dumps(quality))

    def test_stats_quality_signals_reports_provider_adoption_risk(self):
        call_id = str(uuid.uuid4())
        server.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=200,
            output_tokens_est=20,
            actual_input_tokens=200,
            actual_output_tokens=20,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
            routing_json=stable_json({
                "applied": True,
                "policy_source": "local-manual",
                "phase_canary": {
                    "status": "holdout",
                    "cohort": "canary_holdout",
                    "policy_id": "phase-tool-result-haiku",
                },
            }),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="raw-local-session",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_provider_tool_adoption_window(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            updated_at=utc_now(),
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            app_family="claude",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            category="tool-result",
            workflow_phase="tool-result",
            policy_source="local-manual",
            policy_ids_json="[]",
            call_id=call_id,
            fulfilled_call_id=None,
            session_digest="sha256:secret-session-digest",
            correlation_digest="sha256:secret-tool-digest",
            status="abandoned",
            reason="ttl-expired-without-tool-result",
            age_bucket="1_6h",
            tool_use_count=1,
            tool_result_count=0,
            metadata_json=None,
        )

        quality = asyncio.run(stats_views.stats_quality_signals(server.store, limit=5))
        rendered = json.dumps(quality, sort_keys=True)
        item = next(row for row in quality["recent"] if row["unit_id"] == f"provider_call:{call_id}")
        signals = item["quality_signals"]

        self.assertEqual(signals["status"], "success")
        self.assertEqual(signals["risk_level"], "warning")
        self.assertIn("tool-use-abandoned", signals["signal_codes"])
        self.assertIn("optimized-adoption-risk", signals["signal_codes"])
        self.assertEqual(signals["provider_adoption"]["status_counts"]["abandoned"], 1)
        self.assertEqual(signals["optimization_cohorts"][0]["cohort"], "canary_holdout")
        self.assertFalse(quality["privacy"]["raw_tool_payloads_included"])
        self.assertNotIn("secret-tool-digest", rendered)
        self.assertNotIn("secret-session-digest", rendered)
        self.assertNotIn("raw-local-session", rendered)

    def test_provider_adoption_health_reports_applied_holdout_rates_without_raw_ids(self):
        def log_call_with_window(suffix, *, cohort, status, reason, family="phase_routing"):
            call_id = f"provider-adoption-call-secret-{suffix}"
            server.store.log_call(
                id=call_id,
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=200,
                output_tokens_est=20,
                actual_input_tokens=200,
                actual_output_tokens=20,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
                routing_json=stable_json({
                    "applied": cohort == "canary_applied",
                    "policy_source": "local-manual",
                    "phase_canary": {
                        "status": cohort,
                        "cohort": cohort,
                        "policy_id": "phase-tool-result-haiku",
                    },
                }),
                cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw provider adoption prompt secret"}]}),
                response_json=stable_json({"content": "raw provider adoption response secret"}),
                session_id="provider-adoption-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
                requested_model_family="sonnet",
                routed_model_family="haiku",
            )
            server.store.log_provider_tool_adoption_window(
                id=f"provider-adoption-window-secret-{suffix}",
                created_at=utc_now(),
                updated_at=utc_now(),
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
                app_family="claude",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                category="tool-result",
                workflow_phase="tool-result",
                policy_source="local-manual",
                policy_ids_json=stable_json([family]),
                call_id=call_id,
                fulfilled_call_id=call_id if status == "fulfilled" else None,
                session_digest="sha256:provider-adoption-session-digest-secret",
                correlation_digest="sha256:provider-adoption-tool-digest-secret",
                status=status,
                reason=reason,
                age_bucket="0_1m",
                tool_use_count=1,
                tool_result_count=1 if status == "fulfilled" else 0,
                metadata_json=stable_json({"raw_tool_payload": "provider adoption raw tool secret"}),
            )

        log_call_with_window("applied-ok", cohort="canary_applied", status="fulfilled", reason="matched-subsequent-tool-result")
        log_call_with_window("applied-risk", cohort="canary_applied", status="abandoned", reason="ttl-expired-without-tool-result")
        log_call_with_window("holdout-ok", cohort="canary_holdout", status="fulfilled", reason="matched-subsequent-tool-result")

        payload = asyncio.run(stats_views.stats_provider_adoption_health(server.store, limit=20))
        rendered = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["schema"], "agentflow.provider_adoption_dashboard_health.v1")
        self.assertEqual(payload["summary"]["window_count"], 3)
        self.assertEqual(payload["summary"]["fulfilled_count"], 2)
        cohorts = {
            (row["optimization_family"], row["cohort"]): row
            for row in payload["cohort_health"]
        }
        self.assertEqual(cohorts[("phase_routing", "applied")]["window_count"], 2)
        self.assertEqual(cohorts[("phase_routing", "applied")]["fulfilled_count"], 1)
        self.assertEqual(cohorts[("phase_routing", "holdout")]["window_count"], 1)
        self.assertEqual(cohorts[("phase_routing", "holdout")]["adoption_rate"], 1.0)
        comparison = payload["cohort_comparisons"][0]
        self.assertEqual(comparison["optimization_family"], "phase_routing")
        self.assertEqual(comparison["applied_windows"], 2)
        self.assertEqual(comparison["holdout_windows"], 1)
        self.assertEqual(payload["blocker_reason_breakdown"][0]["value"], "ttl-expired-without-tool-result")
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertFalse(payload["privacy"]["raw_tool_payloads_included"])

        for forbidden in (
            "provider-adoption-call-secret",
            "provider-adoption-window-secret",
            "provider-adoption-session-secret",
            "provider-adoption-session-digest-secret",
            "provider-adoption-tool-digest-secret",
            "raw provider adoption prompt secret",
            "raw provider adoption response secret",
            "provider adoption raw tool secret",
            "request_json",
            "response_json",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_provider_adoption_health_endpoint_and_dashboard_are_read_only_metadata(self):
        call_id = "provider-adoption-dashboard-call-secret"
        server.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=200,
            output_tokens_est=20,
            actual_input_tokens=200,
            actual_output_tokens=20,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
            routing_json=stable_json({
                "applied": True,
                "policy_source": "local-manual",
                "phase_canary": {"status": "canary_applied", "cohort": "canary_applied"},
            }),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="provider-adoption-dashboard-session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_provider_tool_adoption_window(
            id="provider-adoption-dashboard-window-secret",
            created_at=utc_now(),
            updated_at=utc_now(),
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            app_family="claude",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            category="tool-result",
            workflow_phase="tool-result",
            policy_source="local-manual",
            policy_ids_json="[]",
            call_id=call_id,
            fulfilled_call_id=call_id,
            session_digest="sha256:provider-adoption-dashboard-session-digest-secret",
            correlation_digest="sha256:provider-adoption-dashboard-tool-digest-secret",
            status="fulfilled",
            reason="matched-subsequent-tool-result",
            age_bucket="0_1m",
            tool_use_count=1,
            tool_result_count=1,
            metadata_json=None,
        )

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream=None,
            limiter_status=lambda: [],
            limiter_config={},
        )
        client = TestClient(app)
        endpoint = client.get("/agentflow/stats/provider-adoption-health?limit=20")
        dashboard = client.get("/agentflow/dashboard")
        rendered = json.dumps(endpoint.json(), sort_keys=True) + dashboard.text

        self.assertEqual(endpoint.status_code, 200)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Provider adoption quality health", dashboard.text)
        self.assertIn("provider-adoption-cohorts-tbody", dashboard.text)
        self.assertIn("fetch('/agentflow/stats/provider-adoption-health?limit=1000')", dashboard.text)
        self.assertNotIn("provider-adoption-dashboard-call-secret", rendered)
        self.assertNotIn("provider-adoption-dashboard-window-secret", rendered)
        self.assertNotIn("provider-adoption-dashboard-session-secret", rendered)
        self.assertNotIn("provider-adoption-dashboard-session-digest-secret", rendered)
        self.assertNotIn("provider-adoption-dashboard-tool-digest-secret", rendered)
        self.assertNotIn("<form", dashboard.text.lower())
        self.assertNotIn("contenteditable", dashboard.text.lower())

    def test_usage_by_owner_groups_provider_calls_and_codex_turns(self):
        old_engineer = os.environ.get("AGENTFLOW_ENGINEER")
        old_app = os.environ.get("AGENTFLOW_APP")
        os.environ["AGENTFLOW_ENGINEER"] = "ada"
        os.environ["AGENTFLOW_APP"] = "code-workbench"
        try:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=20,
                input_tokens_est=1_000,
                output_tokens_est=100,
                actual_input_tokens=1_000,
                actual_output_tokens=100,
                cost_est_usd=0.02,
                cost_baseline_usd=0.025,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"text_chars": 10_000, "category": "tool-result"}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="shared-session",
                category="tool-result",
                cache_creation_input_tokens=1_000,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=200,
                provider="anthropic",
            )
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/responses",
                requested_model="gpt-5-codex",
                routed_model="gpt-5-codex",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=30,
                input_tokens_est=2_000,
                output_tokens_est=300,
                actual_input_tokens=2_000,
                actual_output_tokens=300,
                cost_est_usd=0.03,
                cost_baseline_usd=0.05,
                crunch_json=stable_json({"changed": True}),
                routing_json=stable_json({"text_chars": 8_200, "category": "chat"}),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="shared-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=500,
                retry_count=0,
                thinking_output_tokens=0,
                provider="openai",
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id="req-usage",
                thread_id="thread-usage",
                message_chars=500,
                params_chars=50,
                input_items=2,
                input_text_chars=321,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="shared-session",
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id="req-usage",
                thread_id="thread-usage",
                message_chars=100,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=123,
                error_code=None,
                error_message=None,
                latency_ms=1200,
                session_id="shared-session",
            )

            result = asyncio.run(stats_views.stats_usage_by_owner(server.store))
        finally:
            if old_engineer is None:
                os.environ.pop("AGENTFLOW_ENGINEER", None)
            else:
                os.environ["AGENTFLOW_ENGINEER"] = old_engineer
            if old_app is None:
                os.environ.pop("AGENTFLOW_APP", None)
            else:
                os.environ["AGENTFLOW_APP"] = old_app

        self.assertEqual(result["schema"], "agentflow.usage_by_owner.v1")
        self.assertEqual(result["summary"]["buckets"], 1)
        self.assertFalse(result["summary"]["codex_cost_unknown"])
        self.assertEqual(result["summary"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(result["summary"]["provider_reported_spend_usd"], 0.05, places=6)
        self.assertAlmostEqual(result["summary"]["codex_estimated_spend_usd"], 0.00056, places=6)
        self.assertAlmostEqual(result["summary"]["calculated_spend_usd"], 0.05056, places=6)
        [bucket] = result["buckets"]

        self.assertEqual(bucket["bucket_kind"], "engineer_app")
        self.assertEqual(bucket["bucket_label"], "ada / code-workbench")
        self.assertEqual(bucket["provider_calls"], 2)
        self.assertEqual(bucket["codex_turns"], 1)
        self.assertEqual(bucket["turns"], 3)
        self.assertEqual(bucket["provider_input_tokens"], 4_500)
        self.assertEqual(bucket["provider_output_tokens"], 400)
        self.assertEqual(bucket["provider_total_tokens"], 4_900)
        self.assertEqual(bucket["codex_input_text_chars"], 321)
        self.assertEqual(bucket["codex_result_chars"], 123)
        self.assertEqual(bucket["codex_input_tokens_est"], 80)
        self.assertEqual(bucket["codex_output_tokens_est"], 30)
        self.assertEqual(bucket["codex_total_tokens_est"], 110)
        self.assertEqual(bucket["input_tokens"], 4_580)
        self.assertEqual(bucket["output_tokens"], 430)
        self.assertEqual(bucket["total_tokens"], 5_010)
        self.assertEqual(bucket["token_basis"], "mixed")
        self.assertTrue(bucket["provider_cost_known"])
        self.assertTrue(bucket["codex_cost_known"])
        self.assertTrue(bucket["codex_cost_estimated"])
        self.assertFalse(bucket["excludes_unknown_codex_app_cost"])
        self.assertEqual(bucket["codex_mutation_safe_turns"], 0)
        self.assertEqual(bucket["codex_telemetry_only_turns"], 1)
        self.assertEqual(bucket["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(bucket["codex_cost_est_usd"], 0.00056, places=6)
        self.assertAlmostEqual(bucket["spend_usd"], 0.05056, places=6)
        self.assertAlmostEqual(bucket["baseline_cost_usd"], 0.07556, places=6)
        self.assertAlmostEqual(bucket["cache_savings_usd"], 0.0, places=6)
        self.assertAlmostEqual(bucket["prompt_cache_read_savings_usd"], 0.000563, places=6)
        self.assertAlmostEqual(bucket["captured_savings_usd"], 0.025, places=6)
        self.assertAlmostEqual(bucket["hard_floor_usd"], 0.05056, places=6)
        self.assertEqual(
            {row["source_surface"]: row["units"] for row in bucket["source_surfaces"]},
            {"anthropic_messages": 1, "codex_turn": 1, "openai_responses": 1},
        )
        self.assertEqual(bucket["thinking_tokens"], 200)
        self.assertEqual(bucket["large_tool_result_calls"], 1)
        self.assertGreater(bucket["potential_hint_count"], 0)
        hint_codes = {hint["code"] for hint in bucket["remaining_saving_potential_hints"]}
        self.assertIn("thinking_output", hint_codes)
        self.assertIn("cache_warmup", hint_codes)
        self.assertIn("large_tool_result_context", hint_codes)
        self.assertFalse(result["grouping"]["raw_prompt_logging"])
        self.assertEqual(result["grouping"]["display_name"], "By source")
        self.assertEqual(result["grouping"]["primary_fields"], ["AGENTFLOW_ENGINEER", "AGENTFLOW_APP", "app_family"])
        self.assertEqual(result["grouping"]["fallback_fields"], ["session_id"])
        json.dumps(result)

    def test_dashboard_exposes_unified_recent_calls_table(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Recent calls</button>", html)
        self.assertIn("<h2>Recent calls</h2>", html)
        self.assertIn("id=\"activity-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/activity?limit=100')", html)
        self.assertIn('<th data-sort-type="text">Surface</th>', html)
        self.assertIn('<th data-sort-type="text">Granularity</th>', html)
        self.assertIn('<th data-sort-type="text">App family</th>', html)
        self.assertIn("not provider-replayable", html)
        self.assertIn("Codex estimated from chars", html)
        self.assertNotIn(">Activity</button>", html)
        self.assertNotIn(">Provider calls</button>", html)
        self.assertNotIn(">Codex debug</button>", html)
        self.assertNotIn("id=\"provider-tbody\"", html)
        self.assertNotIn("id=\"codex-tbody\"", html)
        self.assertIn("const operationalTabs=['safety','activity','usage','codex','weekly','categories','cache','errors','limiter','policies','sessions','research']", html)
        self.assertIn("const researchTabs=['adoption','terminal','thinking','scaffold','openai','evalqueue','coordinator','activationnext','promotionblockers','managed','phaserouting','phasememory','oldcontext']", html)
        self.assertIn(">Activation next actions</button>", html)
        self.assertIn("id=\"evidence-activation-summary-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/evidence-to-activation-next-actions?limit=20')", html)

    def test_dashboard_exposes_terminal_output_compaction_readiness_panel(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Terminal compaction</button>", html)
        self.assertIn("<h2>Terminal-output compaction readiness</h2>", html)
        self.assertIn("id=\"terminal-compaction-summary-tbody\"", html)
        self.assertIn("id=\"terminal-compaction-policy-tbody\"", html)
        self.assertIn("id=\"terminal-compaction-candidates-tbody\"", html)
        self.assertIn("id=\"terminal-compaction-impact-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/terminal-output-compaction?opportunity_limit=250&impact_limit=100')", html)
        self.assertIn("terminal text omitted", html)
        self.assertIn("policy contents omitted", html)

    def test_dashboard_exposes_codex_quota_token_usage_panel(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Codex quota</button>", html)
        self.assertIn("<h2>Codex quota and token usage</h2>", html)
        self.assertIn("id=\"codex-quota-tbody\"", html)
        self.assertIn("id=\"codex-rate-scopes-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/codex-effectiveness?limit=500')", html)
        self.assertIn("quota_and_token_usage", html)
        self.assertIn("raw commands omitted", html)
        self.assertIn("raw transcripts omitted", html)

    def test_dashboard_policy_panel_renders_codex_app_surface_state(self):
        html = stats_views.dashboard_html()

        self.assertIn("Codex app-server", html)
        self.assertIn("Codex exact cache off", html)
        self.assertIn("safe keys", html)
        self.assertIn("action-like skip on", html)

    def test_dashboard_exposes_usage_by_source_table(self):
        html = stats_views.dashboard_html()

        self.assertIn(">By source</button>", html)
        self.assertIn("<h2>Usage by source</h2>", html)
        self.assertIn("id=\"usage-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/usage')", html)
        self.assertIn('<th data-sort-type="text">Source</th>', html)
        self.assertIn('<th data-sort-type="text">Grouped by</th>', html)
        self.assertIn('<th data-sort-type="text">Surfaces</th>', html)
        self.assertIn("AGENTFLOW_ENGINEER + AGENTFLOW_APP", html)
        self.assertIn("app_family + session_id", html)
        self.assertIn('<th data-sort-type="number">Turns</th>', html)
        self.assertIn('<th data-sort-type="number">Provider calls</th>', html)
        self.assertIn('<th data-sort-type="number">Codex turns</th>', html)
        self.assertIn("Remaining saving potential", html)
        self.assertIn("Codex estimated", html)
        self.assertNotIn("Codex cost unknown", html)
        self.assertNotIn("Usage by app / engineer", html)
        self.assertNotIn("No app or engineer usage today", html)

    def test_dashboard_exposes_executive_summary_cards(self):
        html = stats_views.dashboard_html()

        self.assertEqual(html.count("class=\"card\""), 2)
        self.assertEqual(html.count("class=\"card green\""), 1)
        self.assertEqual(html.count("class=\"card yellow\""), 0)
        self.assertEqual(html.count("class=\"card blue\""), 1)
        self.assertIn("Tokens today", html)
        self.assertIn("Calculated spend", html)
        self.assertNotIn("Hard floor", html)
        self.assertNotIn("c-floor", html)
        self.assertNotIn("hard_floor_usd", html)
        self.assertNotIn("Ops health", html)
        self.assertIn("Errors today", html)
        self.assertIn("errors_today", html)
        self.assertIn("executive_summary", html)
        self.assertIn("today_buckets", html)
        self.assertIn("Codex estimated", html)
        self.assertNotIn("Calls today", html)
        self.assertNotIn("Saved by routing", html)
        self.assertNotIn("Provider cache discount", html)
        self.assertNotIn("Old-context summaries", html)
        self.assertNotIn("Thinking cost today", html)
        self.assertIn("Codex app-server", html)

    def test_dashboard_exposes_error_breakdown_tables(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Errors</button>", html)
        self.assertIn("<h2>Errors today</h2>", html)
        self.assertIn("<h2>Errors all time</h2>", html)
        self.assertIn("id=\"errors-today-tbody\"", html)
        self.assertIn("id=\"errors-tbody\"", html)
        self.assertIn("today_error_breakdown", html)
        self.assertIn("error_breakdown", html)
        self.assertIn("refreshErrors", html)
        self.assertIn('<th data-sort-type="text">Type</th>', html)
        self.assertIn('<th data-sort-type="number">Status</th>', html)
        self.assertIn('<th data-sort-type="text">Provider</th>', html)
        self.assertIn('<th data-sort-type="text">Tier</th>', html)

    def test_dashboard_exposes_sqlite_maintenance_summary_only(self):
        result = asyncio.run(stats_views.stats_sqlite_maintenance(server.store))
        html = stats_views.dashboard_html()
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream=None,
            limiter_status=lambda: [],
            limiter_config={},
        )
        client = TestClient(app)
        endpoint = client.get("/agentflow/stats/sqlite-maintenance")

        self.assertEqual(result["schema"], "agentflow.sqlite_maintenance_dashboard.v1")
        self.assertEqual(result["summary"]["retention_days"], 7)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertEqual(endpoint.status_code, 200)
        self.assertEqual(endpoint.json()["summary"]["retention_days"], 7)
        self.assertIn("SQLite maintenance", html)
        self.assertIn("sqlite-maintenance-tbody", html)
        self.assertIn("fetch('/agentflow/stats/sqlite-maintenance')", html)
        rendered = json.dumps(endpoint.json(), sort_keys=True)
        self.assertNotIn("request_json", rendered)
        self.assertNotIn("response_json", rendered)
        self.assertNotIn("payload_json", rendered)

    def test_dashboard_tables_are_sortable_and_filterable(self):
        html = stats_views.dashboard_html()

        self.assertIn("function initDataTables", html)
        self.assertIn("function applyDataTableState", html)
        self.assertIn("function applyAllDataTables", html)
        self.assertIn("const tableState={}", html)
        self.assertIn("className='table-filter'", html)
        self.assertIn("setTableSort(table,index)", html)
        self.assertIn("data-sort-type=\"money\"", html)
        self.assertIn("data-sort-type=\"percent\"", html)
        self.assertIn("data-sort-type=\"latency\"", html)
        self.assertIn("data-sort-type=\"time\"", html)
        self.assertIn("<th data-sort-type=\"text\">Surface</th><th data-sort-type=\"text\">App</th><th data-sort-type=\"text\">Session</th>", html)
        self.assertIn("<th data-sort-type=\"number\">Codex turns</th>", html)
        self.assertIn("<th data-sort-type=\"number\">Codex input</th>", html)
        self.assertIn("row.codex_routed_turns", html)
        self.assertIn("No matching rows", html)
        self.assertIn("applyAllDataTables();", html)

        for table_id in (
            "activity",
            "usage",
            "cache-today",
            "cache-all",
            "errors-today",
            "errors-all",
            "codex-quota",
            "codex-rate-scopes",
            "sessions",
        ):
            self.assertIn(f'data-table-id="{table_id}"', html)

    def test_dashboard_coalesces_full_stats_loading(self):
        html = stats_views.dashboard_html()

        self.assertEqual(html.count("fetch('/agentflow/stats/full')"), 1)
        self.assertIn("const FULL_STATS_TTL_MS=5000", html)
        self.assertIn("let fullStatsInFlight=null", html)
        self.assertIn("if(fullStatsInFlight)return fullStatsInFlight", html)
        self.assertEqual(html.count("const d=await loadFullStats();"), 4)
        self.assertIn("async function refresh()", html)
        self.assertIn("async function refreshCategories()", html)
        self.assertIn("async function refreshCache()", html)
        self.assertIn("async function refreshErrors()", html)

    def test_dashboard_polling_is_visibility_and_active_tab_aware(self):
        html = stats_views.dashboard_html()

        self.assertIn("const tabRefreshers=", html)
        self.assertIn("document.addEventListener('visibilitychange'", html)
        self.assertIn("if(document.hidden)return", html)
        self.assertIn("active tab only", html)
        self.assertIn("research on demand", html)
        self.assertIn("research loaded on demand", html)
        self.assertIn("function refreshCurrentPanel()", html)
        self.assertIn("refreshActiveTab({force:true})", html)
        self.assertIn("setInterval(refreshShell,shellMs)", html)
        self.assertIn("setInterval(refreshActiveTab,activeMs)", html)
        self.assertIn("if(isResearchTab(activeTabName)&&!options.force)", html)
        self.assertNotIn("setInterval(refreshTerminalOutputCompaction", html)
        self.assertNotIn("setInterval(refreshRepeatedScaffold", html)
        self.assertNotIn("setInterval(refreshOptimizationCoordinator", html)
        self.assertNotIn("\nrefreshTerminalOutputCompaction();", html)
        self.assertNotIn("\nrefreshRepeatedScaffold();", html)
        self.assertNotIn("\nrefreshOptimizationCoordinator();", html)
        self.assertIn(
            "cache:[refreshCache,refreshOpenAICacheReplayReadiness,refreshOpenAIToolCacheInvalidationBurndown]",
            html,
        )
        self.assertIn("fetch('/agentflow/stats/openai-cache-replay-readiness?opportunity_limit=250&impact_limit=100')", html)
        self.assertIn(
            "fetch('/agentflow/stats/openai-tool-cache-invalidation-burndown?opportunity_limit=250&impact_limit=100&row_limit=25')",
            html,
        )
        self.assertIn("fetch('/agentflow/stats/repeated-scaffold-opportunity?limit=250&min_repeated_rows=2')", html)
        self.assertIn("fetch('/agentflow/stats/optimization-coordinator?limit=250')", html)
        self.assertIn("fetch('/agentflow/stats/local-pattern-coverage?limit=250')", html)

    def test_dashboard_expensive_stats_endpoint_uses_short_ttl_cache(self):
        calls = 0

        async def fake_provider_adoption_health(store_obj, limit=5000):
            nonlocal calls
            calls += 1
            return {
                "schema": "agentflow.provider_adoption_dashboard_health.v1",
                "call_count": calls,
                "limit": limit,
            }

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream=None,
            limiter_status=lambda: [],
            limiter_config={},
        )
        client = TestClient(app)

        with patch("agentflow_proxy.dashboard_app.stats_views.stats_provider_adoption_health", side_effect=fake_provider_adoption_health):
            first = client.get("/agentflow/stats/provider-adoption-health?limit=20")
            second = client.get("/agentflow/stats/provider-adoption-health?limit=20")
            different_query = client.get("/agentflow/stats/provider-adoption-health?limit=21")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(different_query.status_code, 200)
        self.assertEqual(first.json()["call_count"], 1)
        self.assertEqual(second.json()["call_count"], 1)
        self.assertEqual(different_query.json()["call_count"], 2)
        self.assertEqual(calls, 2)

    def test_proxy_dashboard_router_uses_current_store(self):
        response = TestClient(server.app).get("/agentflow/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["calls"], 0)

    def test_sessions_identify_context_plateaus(self):
        text_sizes = [10_000, 10_200, 10_150, 15_000]
        for idx, text_chars in enumerate(text_sizes):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=text_chars // 4,
                output_tokens_est=1,
                actual_input_tokens=text_chars // 4,
                actual_output_tokens=1,
                cost_est_usd=0.01,
                cost_baseline_usd=0.01,
                crunch_json=stable_json({"changed": True, "saved_chars": 100 + idx}),
                routing_json=stable_json({"text_chars": text_chars}),
                cache_json=None,
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-plateau",
                category="tool-heavy",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=1_000 if idx == 0 else 0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]

        self.assertEqual(session["session_id"], "session-plateau")
        self.assertEqual(session["plateau_pairs"], 2)
        self.assertEqual(session["median_text_chars"], 10_175)
        self.assertEqual(session["p90_text_chars"], 15_000)
        self.assertEqual(plateau["session_id"], "session-plateau")
        self.assertEqual(plateau["calls"], 4)
        self.assertEqual(plateau["plateau_pairs"], 2)
        self.assertEqual(plateau["crunch_saved_chars"], 406)
        self.assertAlmostEqual(plateau["cache_read_savings_usd"], 0.0027, places=6)
        self.assertFalse(plateau["flagged"])
        self.assertEqual(result["context_plateau_policy"]["min_text_chars"], 8_000)
        json.dumps(result)

    def test_session_phase_memory_endpoint_and_dashboard_are_metadata_only(self):
        secret_session_ready = "secret-session-ready"
        secret_session_blocked = "secret-session-blocked"
        secret_prompt = "SECRET_PHASE_MEMORY_PROMPT /tmp/private-project/main.py"
        raw_phase_fields = {
            "prompt": secret_prompt,
            "messages": [{"role": "user", "content": "SECRET_PHASE_MEMORY_MESSAGE"}],
            "content": "SECRET_PHASE_MEMORY_CONTENT",
            "tool_payload": {"arguments": "SECRET_PHASE_MEMORY_TOOL_PAYLOAD"},
            "request_id": "req_phase_memory_dashboard_secret",
            "cache_key": "cache-key-phase-memory-dashboard-secret",
            "file_path": "/tmp/private-project/main.py",
            "session_id": "secret-session-dashboard-raw-field",
            "raw_request": {"messages": [{"content": secret_prompt}]},
        }
        for idx in range(3):
            server.store.log_call(
                id=f"phase-ready-{idx}",
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=10,
                input_tokens_est=250,
                output_tokens_est=10,
                actual_input_tokens=250,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.001,
                crunch_json=stable_json({"changed": True, "tokens_saved_est": 1500, **raw_phase_fields}),
                routing_json=stable_json({"workflow_phase": "summary", "text_chars": 1000, **raw_phase_fields}),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss", **raw_phase_fields}),
                error=None,
                request_json=stable_json({"messages": [{"content": secret_prompt}], **raw_phase_fields}),
                response_json=stable_json({"content": [{"text": "SECRET_PHASE_MEMORY_RESPONSE"}]}),
                session_id=secret_session_ready,
                category="short-completion",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )
        for idx, text_chars in enumerate((40_000, 40_500, 40_200)):
            routing = {"category": "tool-result", "text_chars": text_chars, **raw_phase_fields}
            status_code = 200
            retry_count = 0
            if idx == 2:
                routing.update({
                    "fallback_reason": "rate_limited",
                    "session_phase_memory": {
                        "status": "blocked",
                        "reason": "cache-key-phase-memory-dashboard-secret",
                        **raw_phase_fields,
                    },
                })
                status_code = 429
                retry_count = 1
            server.store.log_call(
                id=f"phase-blocked-{idx}",
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=status_code,
                latency_ms=20,
                input_tokens_est=text_chars // 4,
                output_tokens_est=10,
                actual_input_tokens=text_chars // 4,
                actual_output_tokens=10,
                cost_est_usd=0.01,
                cost_baseline_usd=0.02,
                crunch_json=stable_json({"changed": True, "tokens_saved_est": 5000, **raw_phase_fields}),
                routing_json=stable_json(routing),
                cache_json=stable_json(
                    {
                        "status": "skipped",
                        "reason": "cache-key-phase-memory-dashboard-secret",
                        **raw_phase_fields,
                    }
                ),
                error="SECRET_PHASE_MEMORY_ERROR" if status_code >= 400 else None,
                request_json=stable_json({"messages": [{"content": secret_prompt}], **raw_phase_fields}),
                response_json=None,
                session_id=secret_session_blocked,
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=server.store.path,
            upstream="https://anthropic.test",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        client = TestClient(app)
        response = client.get("/agentflow/stats/session-phase-memory")
        dashboard = client.get("/agentflow/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema"], "agentflow.session_phase_memory_dashboard.v1")
        self.assertEqual(payload["summary"]["memory_ready_session_count"], 1)
        self.assertEqual(payload["summary"]["blocked_session_count"], 1)
        self.assertEqual(payload["summary"]["decision_usage"]["decision_count"], 1)
        self.assertEqual(payload["summary"]["decision_usage"]["status_counts"], [{"value": "blocked", "count": 1}])
        self.assertEqual(payload["summary"]["decision_usage"]["reason_counts"], [{"value": "unknown", "count": 1}])
        ready = next(row for row in payload["sessions"] if row["readiness"] == "ready")
        blocked = next(row for row in payload["sessions"] if row["readiness"] == "blocked")
        self.assertEqual(ready["dominant_phase"], "summary")
        self.assertEqual(ready["phase_stability"], 1.0)
        self.assertEqual(ready["model_family_floor"], "sonnet")
        self.assertIn("recent_errors", blocked["blocker_reasons"])
        self.assertIn("recent_retries", blocked["blocker_reasons"])
        self.assertIn("recent_routing_fallback", blocked["blocker_reasons"])
        self.assertEqual(blocked["context_plateau"]["pairs"], 2)
        self.assertFalse(payload["privacy"]["raw_session_ids_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["provider_calls_made"])

        rendered = json.dumps(payload, sort_keys=True) + dashboard.text
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("/agentflow/stats/session-phase-memory", dashboard.text)
        self.assertIn("Session phase memory readiness", dashboard.text)
        self.assertIn("session-phase-memory-summary-tbody", dashboard.text)
        self.assertIn("session-phase-memory-sessions-tbody", dashboard.text)
        self.assertNotIn(secret_session_ready, rendered)
        self.assertNotIn(secret_session_blocked, rendered)
        self.assertNotIn(secret_prompt, rendered)
        self.assertNotIn("SECRET_PHASE_MEMORY_RESPONSE", rendered)
        self.assertNotIn("SECRET_PHASE_MEMORY_ERROR", rendered)
        self.assertNotIn("SECRET_PHASE_MEMORY_MESSAGE", rendered)
        self.assertNotIn("SECRET_PHASE_MEMORY_CONTENT", rendered)
        self.assertNotIn("SECRET_PHASE_MEMORY_TOOL_PAYLOAD", rendered)
        self.assertNotIn("req_phase_memory_dashboard_secret", rendered)
        self.assertNotIn("cache-key-phase-memory-dashboard-secret", rendered)
        self.assertNotIn("secret-session-dashboard-raw-field", rendered)
        self.assertNotIn("/tmp/private-project", rendered)
        self.assertNotIn("<form", dashboard.text.lower())
        self.assertNotIn("contenteditable", dashboard.text.lower())

    def test_limiter_stats_include_active_cooldown_and_recent_rate_limits(self):
        server._tier_backoff_until.clear()
        server._tier_backoff_until["haiku"] = time.time() + 90
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=429,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error="temporarily limiting requests for haiku tier; retry after 90s",
            request_json=None,
            response_json=None,
            session_id="session-limiter",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=429,
            latency_ms=10,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error="upstream_error: status=429",
            request_json=None,
            response_json=None,
            session_id="session-limiter",
            category="tool-heavy",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=3,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_limiter(server.store, server._tier_backoff_status, server._dashboard_limiter_config()))
        tiers = {row["tier"]: row for row in result["tiers"]}

        self.assertTrue(tiers["haiku"]["active"])
        self.assertGreater(tiers["haiku"]["seconds_remaining"], 0)
        self.assertIsNotNone(tiers["haiku"]["cooldown_until"])
        self.assertEqual(tiers["haiku"]["max_concurrent"], server.MAX_CONCURRENT_PER_TIER)
        self.assertIsNotNone(tiers["sonnet"]["last_upstream_429_at"])
        self.assertEqual(result["summary"]["active_cooldowns"], 1)
        self.assertEqual(result["summary"]["local_throttled_recent"], 1)
        self.assertEqual(result["summary"]["upstream_limited_recent"], 1)
        self.assertEqual(result["recent_rate_limits"][0]["tier"], "sonnet")
        self.assertEqual(result["recent_rate_limits"][1]["tier"], "haiku")
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
