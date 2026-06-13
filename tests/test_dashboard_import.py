import sys
import asyncio
import importlib.util
import os
import json
import time
import uuid
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("fastapi", "httpx")
)

if HAS_RUNTIME_DEPS:
    import httpx
    from fastapi.testclient import TestClient

    import agentflow_proxy.dashboard_app as dashboard_app
    from agentflow_proxy.dashboard_app import create_dashboard_app
    from agentflow_proxy.store import Store, stable_json, utc_now


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class DashboardImportTests(unittest.TestCase):
    def test_dashboard_import_does_not_import_provider_server(self):
        old_dashboard = sys.modules.pop("agentflow_proxy.dashboard", None)
        old_dashboard_app = sys.modules.pop("agentflow_proxy.dashboard_app", None)
        old_server = sys.modules.pop("agentflow_proxy.server", None)
        old_provider_handlers = sys.modules.pop("agentflow_proxy.provider_handlers", None)
        try:
            import agentflow_proxy.dashboard  # noqa: F401

            self.assertNotIn("agentflow_proxy.server", sys.modules)
            self.assertNotIn("agentflow_proxy.provider_handlers", sys.modules)
        finally:
            sys.modules.pop("agentflow_proxy.dashboard", None)
            sys.modules.pop("agentflow_proxy.dashboard_app", None)
            if old_dashboard is not None:
                sys.modules["agentflow_proxy.dashboard"] = old_dashboard
            if old_dashboard_app is not None:
                sys.modules["agentflow_proxy.dashboard_app"] = old_dashboard_app
            if old_server is not None:
                sys.modules["agentflow_proxy.server"] = old_server
            if old_provider_handlers is not None:
                sys.modules["agentflow_proxy.provider_handlers"] = old_provider_handlers

    def test_dashboard_app_uses_injected_store_and_preserves_routes(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        store = Store(tmp.name)
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event("validate", ok=False, details={"source": "test", "error_count": 1})
            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
            )
            client = TestClient(app)

            health = client.get("/health")
            stats = client.get("/agentflow/stats")
            policies = client.get("/agentflow/stats/policies")
            policy_workbench = client.get("/agentflow/stats/policy-workbench")
            policy_events = client.get("/agentflow/stats/policy-events")
            codex_effectiveness = client.get("/agentflow/stats/codex-effectiveness")
            codex_readiness = client.get("/agentflow/stats/codex-readiness")
            codex_canary_impact = client.get("/agentflow/stats/codex-canary-impact")
            openai_scoreboard = client.get("/agentflow/stats/openai-scoreboard")
            managed_openai_activation = client.get("/agentflow/stats/managed-openai-activation")
            openai_optimization_readiness = client.get("/agentflow/stats/openai-optimization-readiness")
            openai_canary_readiness = client.get("/agentflow/stats/openai-canary-readiness")
            claude_canary_impact = client.get("/agentflow/stats/claude-canary-impact")
            claude_routing_funnel = client.get("/agentflow/stats/claude-routing-promotion-funnel")
            openai_old_context_summary = client.get("/agentflow/stats/openai-old-context-summary")
            openai_cache_replay_readiness = client.get("/agentflow/stats/openai-cache-replay-readiness")
            cache_replay_activation_health = client.get("/agentflow/stats/cache-replay-activation-health")
            terminal_output_compaction = client.get("/agentflow/stats/terminal-output-compaction")
            repeated_scaffold_opportunity = client.get("/agentflow/stats/repeated-scaffold-opportunity")
            instruction_dedup_opportunity = client.get("/agentflow/stats/instruction-dedup-opportunity")
            instruction_dedup_impact = client.get("/agentflow/stats/instruction-dedup-impact")
            repeated_scaffold_impact = client.get("/agentflow/stats/repeated-scaffold-impact")
            repeated_scaffold_activation = client.get("/agentflow/stats/repeated-scaffold-activation")
            scaffold_rollout_health = client.get("/agentflow/stats/scaffold-rollout-health")
            optimization_eval_queue = client.get("/agentflow/stats/optimization-eval-queue")
            optimization_coordinator = client.get("/agentflow/stats/optimization-coordinator")
            optimization_promotion_funnel = client.get("/agentflow/stats/optimization-promotion-funnel")
            rollout_readiness = client.get("/agentflow/stats/rollout-actions/readiness")
            local_pattern_coverage = client.get("/agentflow/stats/local-pattern-coverage")
            phase_routing = client.get("/agentflow/stats/phase-routing")
            safety = client.get("/agentflow/stats/safety")
            admin_reload = client.post("/agentflow/admin/reload-policies")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["mode"], "dashboard-read-only")
            self.assertEqual(stats.status_code, 200)
            self.assertEqual(stats.json()["db"], tmp.name)
            self.assertEqual(stats.json()["calls"], 0)
            self.assertEqual(policies.status_code, 200)
            self.assertEqual(policy_workbench.status_code, 200)
            self.assertEqual(policy_workbench.json()["schema"], "agentflow.policy_workbench_readiness.v1")
            self.assertTrue(policy_workbench.json()["read_only"])
            self.assertFalse(policy_workbench.json()["mutating_dashboard_endpoints"])
            self.assertFalse(policy_workbench.json()["privacy"]["dashboard_mutations_available"])
            self.assertFalse(policy_workbench.json()["privacy"]["provider_calls_made"])
            self.assertEqual(policy_events.status_code, 200)
            self.assertEqual(policy_events.json()["schema"], "agentflow.policy_events.v1")
            self.assertEqual(policy_events.json()["events"][0]["action"], "validate")
            self.assertEqual(codex_effectiveness.status_code, 200)
            self.assertEqual(codex_effectiveness.json()["schema"], "agentflow.codex_app_effectiveness.v1")
            self.assertFalse(codex_effectiveness.json()["privacy"]["raw_prompts_included"])
            self.assertEqual(codex_readiness.status_code, 200)
            self.assertEqual(codex_readiness.json()["schema"], "agentflow.codex_optimization_readiness.v1")
            self.assertFalse(codex_readiness.json()["privacy"]["raw_prompts_included"])
            self.assertFalse(codex_readiness.json()["privacy"]["request_ids_included"])
            self.assertEqual(codex_canary_impact.status_code, 200)
            self.assertEqual(codex_canary_impact.json()["schema"], "agentflow.codex_app_canary_impact_by_rule.v1")
            self.assertFalse(codex_canary_impact.json()["privacy"]["request_ids_included"])
            self.assertEqual(openai_scoreboard.status_code, 200)
            self.assertEqual(openai_scoreboard.json()["schema"], "agentflow.openai_optimization_scoreboard.v1")
            self.assertFalse(openai_scoreboard.json()["privacy"]["provider_calls_made"])
            self.assertEqual(managed_openai_activation.status_code, 200)
            self.assertEqual(managed_openai_activation.json()["schema"], "agentflow.managed_openai_activation.v1")
            self.assertTrue(managed_openai_activation.json()["read_only"])
            self.assertFalse(managed_openai_activation.json()["privacy"]["provider_calls_made"])
            self.assertFalse(managed_openai_activation.json()["privacy"]["request_ids_included"])
            self.assertFalse(managed_openai_activation.json()["privacy"]["cache_keys_included"])
            self.assertEqual(openai_optimization_readiness.status_code, 200)
            self.assertEqual(openai_optimization_readiness.json()["schema"], "agentflow.openai_optimization_readiness.v1")
            self.assertTrue(openai_optimization_readiness.json()["read_only"])
            self.assertFalse(openai_optimization_readiness.json()["privacy"]["provider_calls_made"])
            self.assertFalse(openai_optimization_readiness.json()["privacy"]["request_ids_included"])
            self.assertEqual(openai_canary_readiness.status_code, 200)
            self.assertEqual(openai_canary_readiness.json()["schema"], "agentflow.openai_canary_readiness.v1")
            self.assertEqual(openai_canary_readiness.json()["state"], "disabled")
            self.assertFalse(openai_canary_readiness.json()["privacy"]["provider_calls_made"])
            self.assertEqual(claude_canary_impact.status_code, 200)
            self.assertEqual(claude_canary_impact.json()["schema"], "agentflow.claude_canary_impact.v1")
            self.assertFalse(claude_canary_impact.json()["privacy"]["provider_calls_made"])
            self.assertFalse(claude_canary_impact.json()["privacy"]["request_ids_included"])
            self.assertEqual(claude_routing_funnel.status_code, 200)
            self.assertEqual(claude_routing_funnel.json()["schema"], "agentflow.claude_routing_promotion_funnel.v1")
            self.assertTrue(claude_routing_funnel.json()["read_only"])
            self.assertFalse(claude_routing_funnel.json()["privacy"]["provider_calls_made"])
            self.assertFalse(claude_routing_funnel.json()["privacy"]["request_ids_included"])
            self.assertEqual(openai_old_context_summary.status_code, 200)
            self.assertEqual(openai_old_context_summary.json()["schema"], "agentflow.openai_old_context_summary_opportunity.v1")
            self.assertFalse(openai_old_context_summary.json()["local_policy"]["enabled"])
            self.assertFalse(openai_old_context_summary.json()["local_policy"]["rule_file"]["rule_path_included"])
            self.assertFalse(openai_old_context_summary.json()["privacy"]["provider_calls_made"])
            self.assertFalse(openai_old_context_summary.json()["privacy"]["raw_request_bodies_included"])
            self.assertEqual(openai_cache_replay_readiness.status_code, 200)
            self.assertEqual(openai_cache_replay_readiness.json()["schema"], "agentflow.openai_cache_replay_readiness.v1")
            self.assertFalse(openai_cache_replay_readiness.json()["privacy"]["provider_calls_made"])
            self.assertFalse(openai_cache_replay_readiness.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(openai_cache_replay_readiness.json()["privacy"]["cache_keys_included"])
            self.assertEqual(cache_replay_activation_health.status_code, 200)
            self.assertEqual(cache_replay_activation_health.json()["schema"], "agentflow.cache_replay_activation_health.v1")
            self.assertTrue(cache_replay_activation_health.json()["read_only"])
            self.assertFalse(cache_replay_activation_health.json()["privacy"]["provider_calls_made"])
            self.assertFalse(cache_replay_activation_health.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(cache_replay_activation_health.json()["privacy"]["cache_keys_included"])
            self.assertEqual(terminal_output_compaction.status_code, 200)
            self.assertEqual(terminal_output_compaction.json()["schema"], "agentflow.terminal_output_compaction_readiness.v1")
            self.assertTrue(terminal_output_compaction.json()["read_only"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["provider_calls_made"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["raw_terminal_lines_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["tool_payloads_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["request_ids_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["session_ids_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["cache_keys_included"])
            self.assertFalse(terminal_output_compaction.json()["privacy"]["policy_file_contents_included"])
            self.assertEqual(repeated_scaffold_opportunity.status_code, 200)
            self.assertEqual(repeated_scaffold_opportunity.json()["schema"], "agentflow.repeated_scaffold_opportunity.v1")
            self.assertEqual(repeated_scaffold_opportunity.json()["summary"]["candidate_count"], 0)
            self.assertFalse(repeated_scaffold_opportunity.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(repeated_scaffold_opportunity.json()["privacy"]["request_ids_included"])
            self.assertFalse(repeated_scaffold_opportunity.json()["privacy"]["session_ids_included"])
            self.assertFalse(repeated_scaffold_opportunity.json()["privacy"]["cache_keys_included"])
            self.assertEqual(instruction_dedup_opportunity.status_code, 200)
            self.assertEqual(instruction_dedup_opportunity.json()["schema"], "agentflow.instruction_dedup_opportunity.v1")
            self.assertEqual(instruction_dedup_opportunity.json()["summary"]["candidate_count"], 0)
            self.assertTrue(instruction_dedup_opportunity.json()["read_only"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["raw_instruction_text_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["tool_payloads_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["request_ids_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["session_ids_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["thread_ids_included"])
            self.assertFalse(instruction_dedup_opportunity.json()["privacy"]["cache_keys_included"])
            self.assertEqual(instruction_dedup_impact.status_code, 200)
            self.assertEqual(instruction_dedup_impact.json()["schema"], "agentflow.instruction_dedup_impact.v1")
            self.assertEqual(instruction_dedup_impact.json()["status"], "no-instruction-dedup-canary-metadata")
            self.assertTrue(instruction_dedup_impact.json()["read_only"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["raw_instruction_text_included"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["tool_payloads_included"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["request_ids_included"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["session_ids_included"])
            self.assertFalse(instruction_dedup_impact.json()["privacy"]["cache_keys_included"])
            self.assertFalse(instruction_dedup_impact.json()["managed_lifecycle_feedback_queue"]["privacy"]["payload_json_included"])
            self.assertEqual(repeated_scaffold_impact.status_code, 200)
            self.assertEqual(repeated_scaffold_impact.json()["schema"], "agentflow.repeated_scaffold_impact.v1")
            self.assertEqual(repeated_scaffold_impact.json()["status"], "no-repeated-scaffold-canary-metadata")
            self.assertEqual(
                repeated_scaffold_impact.json()["managed_lifecycle_feedback_queue"]["schema"],
                "agentflow.repeated_scaffold_lifecycle_feedback_queue_status.v1",
            )
            self.assertFalse(repeated_scaffold_impact.json()["privacy"]["provider_calls_made"])
            self.assertFalse(repeated_scaffold_impact.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(repeated_scaffold_impact.json()["privacy"]["request_ids_included"])
            self.assertFalse(repeated_scaffold_impact.json()["privacy"]["session_ids_included"])
            self.assertFalse(repeated_scaffold_impact.json()["privacy"]["cache_keys_included"])
            self.assertFalse(repeated_scaffold_impact.json()["managed_lifecycle_feedback_queue"]["privacy"]["payload_json_included"])
            self.assertEqual(repeated_scaffold_activation.status_code, 200)
            self.assertEqual(repeated_scaffold_activation.json()["schema"], "agentflow.repeated_scaffold_activation.v1")
            self.assertEqual(repeated_scaffold_activation.json()["status"], "no-activation-metadata")
            self.assertFalse(repeated_scaffold_activation.json()["privacy"]["provider_calls_made"])
            self.assertFalse(repeated_scaffold_activation.json()["privacy"]["raw_request_bodies_included"])
            self.assertFalse(repeated_scaffold_activation.json()["privacy"]["optimization_unit_ids_included"])
            self.assertFalse(repeated_scaffold_activation.json()["privacy"]["feedback_payloads_included"])
            self.assertEqual(scaffold_rollout_health.status_code, 200)
            self.assertEqual(scaffold_rollout_health.json()["schema"], "agentflow.scaffold_rollout_health.v1")
            self.assertTrue(scaffold_rollout_health.json()["read_only"])
            self.assertFalse(scaffold_rollout_health.json()["privacy"]["provider_calls_made"])
            self.assertFalse(scaffold_rollout_health.json()["privacy"]["raw_action_payloads_included"])
            self.assertFalse(scaffold_rollout_health.json()["privacy"]["yaml_contents_included"])
            self.assertEqual(optimization_eval_queue.status_code, 200)
            self.assertEqual(optimization_eval_queue.json()["schema"], "agentflow.optimization_eval_queue.v1")
            self.assertFalse(optimization_eval_queue.json()["privacy"]["provider_calls_made"])
            self.assertEqual(optimization_coordinator.status_code, 200)
            self.assertEqual(optimization_coordinator.json()["schema"], "agentflow.optimization_coordinator_dashboard.v1")
            self.assertTrue(optimization_coordinator.json()["read_only"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["provider_calls_made"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["raw_prompts_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["raw_responses_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["provider_bodies_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["file_paths_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["cache_keys_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["request_ids_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["session_ids_included"])
            self.assertFalse(optimization_coordinator.json()["privacy"]["policy_file_contents_included"])
            self.assertEqual(optimization_promotion_funnel.status_code, 200)
            self.assertEqual(optimization_promotion_funnel.json()["schema"], "agentflow.optimization_promotion_funnel.v1")
            self.assertFalse(optimization_promotion_funnel.json()["privacy"]["provider_calls_made"])
            self.assertEqual(rollout_readiness.status_code, 200)
            self.assertEqual(rollout_readiness.json()["schema"], "agentflow.rollout_actions_readiness.v1")
            self.assertFalse(rollout_readiness.json()["privacy"]["raw_action_payloads_included"])
            self.assertEqual(local_pattern_coverage.status_code, 200)
            self.assertEqual(local_pattern_coverage.json()["schema"], "agentflow.local_pattern_coverage.v1")
            self.assertFalse(local_pattern_coverage.json()["privacy"]["raw_prompts_included"])
            self.assertEqual(phase_routing.status_code, 200)
            self.assertEqual(phase_routing.json()["schema"], "agentflow.phase_routing_dashboard.v1")
            self.assertFalse(phase_routing.json()["privacy"]["raw_prompts_included"])
            self.assertEqual(safety.status_code, 200)
            self.assertEqual(safety.json()["schema"], "agentflow.safety_privacy.v1")
            self.assertFalse(safety.json()["privacy"]["raw_prompts_included"])
            policy_json = policies.json()
            self.assertEqual(policy_json["schema"], "agentflow.policy_state.v1")
            self.assertIn("summary", policy_json)
            self.assertIn("workbench", policy_json)
            self.assertEqual(policy_json["workbench"]["schema"], "agentflow.policy_workbench_readiness.v1")
            self.assertFalse(policy_json["workbench"]["privacy"]["raw_prompts_included"])
            self.assertIn("reload_required", policy_json["summary"])
            self.assertIn("reload_required_sections", policy_json["summary"])
            self.assertEqual(policy_json["summary"]["policy_count"], 5)
            self.assertIn("routing", policy_json)
            self.assertIn("crunch", policy_json)
            self.assertIn("cache", policy_json)
            self.assertIn("routing_experiments", policy_json)
            self.assertIn("codex_app", policy_json)
            self.assertIn("policy_source", policy_json["routing"])
            self.assertIn("rule_path", policy_json["routing"])
            self.assertIn("file", policy_json["routing"])
            self.assertIn("reload_required", policy_json["routing"]["file"])
            self.assertIn("policy_source", policy_json["crunch"])
            self.assertIn("rule_path", policy_json["crunch"])
            self.assertIn("file", policy_json["crunch"])
            self.assertIn("reload_required", policy_json["crunch"]["file"])
            self.assertIn("policy_source", policy_json["cache"])
            self.assertIn("rule_path", policy_json["cache"])
            self.assertIn("file", policy_json["cache"])
            self.assertIn("reload_required", policy_json["cache"]["file"])
            self.assertIn("policy_source", policy_json["routing_experiments"])
            self.assertIn("rule_path", policy_json["routing_experiments"])
            self.assertIn("file", policy_json["routing_experiments"])
            self.assertIn("reload_required", policy_json["routing_experiments"]["file"])
            self.assertEqual(policy_json["codex_app"]["surface"], "codex_turn")
            self.assertFalse(policy_json["codex_app"]["review_only"])
            self.assertIn("policy_source", policy_json["codex_app"])
            self.assertIn("rule_path", policy_json["codex_app"])
            self.assertIn("file", policy_json["codex_app"])
            self.assertIn("reload_required", policy_json["codex_app"]["file"])
            self.assertEqual(admin_reload.status_code, 404)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("AgentFlow", dashboard.text)
            self.assertIn("Policies", dashboard.text)
            self.assertIn("/agentflow/stats/policies", dashboard.text)
            self.assertIn("/agentflow/stats/policy-workbench", dashboard.text)
            self.assertIn("/agentflow/stats/policy-events", dashboard.text)
            self.assertIn("Policy workbench readiness", dashboard.text)
            self.assertIn("policy-workbench-tbody", dashboard.text)
            self.assertIn("Policy workbench events", dashboard.text)
            self.assertIn("policy-workbench-events-tbody", dashboard.text)
            self.assertIn("routing-experiment-eligibility-tbody", dashboard.text)
            self.assertIn("Policy reload summary", dashboard.text)
            self.assertIn("policy-summary-tbody", dashboard.text)
            self.assertIn("Codex rules", dashboard.text)
            self.assertIn("/agentflow/stats/codex-readiness", dashboard.text)
            self.assertIn("Codex optimization readiness", dashboard.text)
            self.assertIn("codex-readiness-tbody", dashboard.text)
            self.assertIn("Codex exact-cache canary impact", dashboard.text)
            self.assertIn("codex-cache-readiness-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/codex-canary-impact", dashboard.text)
            self.assertIn("Codex canary impact by rule", dashboard.text)
            self.assertIn("codex-canary-impact-tbody", dashboard.text)
            self.assertIn("Recent policy events", dashboard.text)
            self.assertIn("Safety / privacy status", dashboard.text)
            self.assertIn("/agentflow/stats/safety", dashboard.text)
            self.assertIn("safety-warnings-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/rollout-actions/readiness", dashboard.text)
            self.assertIn("Rollout-action readiness", dashboard.text)
            self.assertIn("rollout-readiness-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/local-pattern-coverage", dashboard.text)
            self.assertIn("Local pattern coverage", dashboard.text)
            self.assertIn("local-pattern-coverage-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/phase-routing", dashboard.text)
            self.assertIn("Phase-routing rollout health", dashboard.text)
            self.assertIn("phase-routing-health-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/optimization-eval-queue", dashboard.text)
            self.assertIn("Optimization eval and promotion candidates", dashboard.text)
            self.assertIn("optimization-eval-candidates-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/optimization-coordinator", dashboard.text)
            self.assertIn("Cross-family coordinator state", dashboard.text)
            self.assertIn("optimization-coordinator-summary-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/optimization-promotion-funnel", dashboard.text)
            self.assertIn("Optimization promotion canary impact", dashboard.text)
            self.assertIn("optimization-promotion-funnel-candidates-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/claude-routing-promotion-funnel", dashboard.text)
            self.assertIn("Claude routing promotion funnel", dashboard.text)
            self.assertIn("claude-routing-funnel-candidates-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/openai-optimization-readiness", dashboard.text)
            self.assertIn("/agentflow/stats/managed-openai-activation", dashboard.text)
            self.assertIn("Managed OpenAI activation", dashboard.text)
            self.assertIn("managed-openai-activation-tbody", dashboard.text)
            self.assertIn("managed-openai-activation-families-tbody", dashboard.text)
            self.assertIn("OpenAI optimization readiness", dashboard.text)
            self.assertIn("openai-optimization-readiness-summary-tbody", dashboard.text)
            self.assertIn("openai-optimization-readiness-families-tbody", dashboard.text)
            self.assertIn("openai-optimization-readiness-conflicts-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/openai-canary-readiness", dashboard.text)
            self.assertIn("OpenAI local canary readiness", dashboard.text)
            self.assertIn("openai-canary-readiness-summary-tbody", dashboard.text)
            self.assertIn("openai-canary-readiness-candidates-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/openai-old-context-summary", dashboard.text)
            self.assertIn("OpenAI old-context summary readiness", dashboard.text)
            self.assertIn("openai-old-context-summary-readiness-tbody", dashboard.text)
            self.assertIn("OpenAI old-context summary endpoint impact", dashboard.text)
            self.assertIn("openai-old-context-summary-groups-tbody", dashboard.text)
            self.assertIn("OpenAI old-context summary quality gates", dashboard.text)
            self.assertIn("openai-old-context-summary-quality-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/openai-cache-replay-readiness", dashboard.text)
            self.assertIn("OpenAI cache replay readiness", dashboard.text)
            self.assertIn("openai-cache-replay-readiness-tbody", dashboard.text)
            self.assertIn("OpenAI cache replay impact gates", dashboard.text)
            self.assertIn("openai-cache-replay-impact-gates-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/cache-replay-activation-health", dashboard.text)
            self.assertIn("Cache replay activation health", dashboard.text)
            self.assertIn("cache-replay-activation-health-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/repeated-scaffold-opportunity", dashboard.text)
            self.assertIn("/agentflow/stats/repeated-scaffold-impact", dashboard.text)
            self.assertIn("/agentflow/stats/repeated-scaffold-activation", dashboard.text)
            self.assertIn("/agentflow/stats/scaffold-rollout-health", dashboard.text)
            self.assertIn("Scaffold crunch", dashboard.text)
            self.assertIn("Managed scaffold rollout", dashboard.text)
            self.assertIn("scaffold-rollout-health-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold crunch readiness", dashboard.text)
            self.assertIn("repeated-scaffold-readiness-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold policy-decision activation", dashboard.text)
            self.assertIn("repeated-scaffold-activation-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold activation groups", dashboard.text)
            self.assertIn("repeated-scaffold-activation-groups-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold opportunity candidates", dashboard.text)
            self.assertIn("repeated-scaffold-opportunities-tbody", dashboard.text)
            self.assertIn("No repeated-scaffold opportunity candidates yet", dashboard.text)
            self.assertIn("Repeated-scaffold canary impact", dashboard.text)
            self.assertIn("repeated-scaffold-impact-summary-tbody", dashboard.text)
            self.assertIn("repeated-scaffold-feedback-queue-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold promotion gates", dashboard.text)
            self.assertIn("repeated-scaffold-impact-candidates-tbody", dashboard.text)
            self.assertIn("No repeated-scaffold canary impact metadata yet", dashboard.text)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()

    def test_repeated_scaffold_dashboard_endpoints_are_content_free(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        policy_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        old_canary_policy = os.environ.get("AGENTFLOW_SCAFFOLD_CANARY_POLICY")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        os.environ["AGENTFLOW_SCAFFOLD_CANARY_POLICY"] = str(Path(policy_tmp.name) / "scaffold_canary_policy.yaml")
        store = Store(tmp.name)

        def log_call(
            *,
            request_json=None,
            crunch_json=None,
            routing_extra=None,
            category="tool-result",
            created_at="2026-06-12T00:00:00+00:00",
        ):
            routing = {
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": category,
                "workflow_phase": "tool-execution",
                "text_chars": 48000,
                "has_tools": category.startswith("tool"),
            }
            if routing_extra:
                routing.update(routing_extra)
            store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=12000,
                output_tokens_est=100,
                actual_input_tokens=12000,
                actual_output_tokens=100,
                cost_est_usd=0.04,
                cost_baseline_usd=0.04,
                crunch_json=stable_json(crunch_json or {"changed": False, "tokens_saved_est": 0}),
                routing_json=stable_json(routing),
                cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
                error=None,
                request_json=stable_json(request_json) if request_json is not None else None,
                response_json=stable_json({"text": "raw response must not leak"}),
                session_id="raw-session-must-not-leak",
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
                requested_model_family="sonnet",
                routed_model_family="sonnet",
            )

        repeated_text = (
            "Repeated provider scaffold line with stable tool framing and operational instructions "
            "that should be counted but never displayed as raw dashboard content."
        )
        raw_body = {
            "model": "claude-sonnet-4-6",
            "system": repeated_text,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": repeated_text + " raw secret one"}]},
                {"role": "assistant", "content": [{"type": "text", "text": repeated_text + " raw secret two"}]},
            ],
            "request_id": "raw-request-id-must-not-leak",
            "cache_key": "raw-cache-key-must-not-leak",
        }
        log_call(request_json=raw_body)
        log_call(request_json=raw_body)

        def repeated_scaffold_crunch(cohort, tokens_saved=1000):
            applied = cohort == "canary_applied"
            holdout = cohort == "canary_holdout"
            canary = {
                "enabled": True,
                "selected": applied,
                "cohort": cohort,
                "fraction": 0.10,
                "unit": "request_fingerprint",
            }
            rule = {
                "rule_id": "reviewed-provider-scaffold",
                "candidate_id": "candidate-provider-123",
                "enabled": True,
                "policy_source": "managed-recommended",
                "applied_count": 1 if applied else 0,
                "holdout_count": 1 if holdout else 0,
                "saved_chars": tokens_saved * 4 if applied else 0,
                "canary": canary,
                "skip_reasons": [{"reason": "canary_holdout", "count": 1, "canary": canary}] if holdout else [],
            }
            return {
                "changed": applied,
                "repeated_provider_scaffolding": {
                    "schema": "agentflow.repeated_provider_scaffolding.v1",
                    "enabled": True,
                    "status": "applied" if applied else "skipped",
                    "reason": "repeated-provider-scaffolding-crunched" if applied else "canary_holdout",
                    "policy_source": "managed-recommended",
                    "category": "tool-result",
                    "saved_chars": tokens_saved * 4 if applied else 0,
                    "tokens_saved_est": tokens_saved if applied else 0,
                    "rules": [rule],
                    "raw_text_included": False,
                    "raw_hashes_included": False,
                },
            }

        managed_repeated_scaffold = {
            "managed_recommendation": {
                "enabled": True,
                "status": "received",
                "reason": "managed repeated scaffold canary",
                "policy_id": "managed-scaffold-dashboard",
                "optimization_unit_id": 77,
                "applied": False,
                "apply_reason": "missing-target-model",
                "crunch": {
                    "profile": "managed",
                    "repeated_provider_scaffolding": {
                        "enabled": True,
                        "rules": [{"id": "managed-dashboard-rule-must-not-leak"}],
                    },
                },
                "outcome_feedback": {
                    "enabled": True,
                    "status": "sent",
                    "reason": "accepted",
                    "optimization_unit_id": 77,
                },
            }
        }
        log_call(
            crunch_json=repeated_scaffold_crunch("canary_applied", 1100),
            routing_extra=managed_repeated_scaffold,
            created_at="2026-06-12T00:01:00+00:00",
        )
        log_call(
            crunch_json=repeated_scaffold_crunch("canary_applied", 1200),
            routing_extra=managed_repeated_scaffold,
            created_at="2026-06-12T00:02:00+00:00",
        )
        log_call(
            crunch_json=repeated_scaffold_crunch("canary_holdout", 0),
            routing_extra=managed_repeated_scaffold,
            created_at="2026-06-12T00:03:00+00:00",
        )
        Path(os.environ["AGENTFLOW_SCAFFOLD_CANARY_POLICY"]).write_text(
            "\n".join([
                "schema: agentflow.scaffold_canary_policy.v1",
                "policy_source: managed-recommended",
                "repeated_provider_scaffolding:",
                "  enabled: true",
                "  rules:",
                "    - id: reviewed-provider-scaffold",
                "      enabled: true",
                "    - id: reviewed-provider-scaffold-second",
                "      enabled: true",
            ]),
            encoding="utf-8",
        )
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "scaffold-rollout-actions-review",
            ok=True,
            details={
                "source": "cli",
                "url": "https://managed.example/scaffold-rollout-actions",
                "fetch_status": "ok",
                "action_count": 3,
                "accepted_action_count": 2,
                "provenance_status": "verified",
                "exit_code": 0,
            },
        )

        try:
            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)

            opportunity_response = client.get("/agentflow/stats/repeated-scaffold-opportunity?limit=20")
            impact_response = client.get("/agentflow/stats/repeated-scaffold-impact?limit=20")
            activation_response = client.get("/agentflow/stats/repeated-scaffold-activation?limit=20")
            rollout_health_response = client.get("/agentflow/stats/scaffold-rollout-health?limit=20")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(opportunity_response.status_code, 200)
            opportunity = opportunity_response.json()
            self.assertEqual(opportunity["schema"], "agentflow.repeated_scaffold_opportunity.v1")
            self.assertGreaterEqual(opportunity["summary"]["candidate_count"], 1)
            self.assertGreater(opportunity["summary"]["projected_saved_tokens"], 0)
            self.assertGreaterEqual(opportunity["candidates"][0]["matched_count"], 2)
            self.assertFalse(opportunity["privacy"]["raw_request_bodies_included"])
            self.assertFalse(opportunity["privacy"]["request_ids_included"])
            self.assertFalse(opportunity["privacy"]["session_ids_included"])
            self.assertFalse(opportunity["privacy"]["cache_keys_included"])

            self.assertEqual(impact_response.status_code, 200)
            impact = impact_response.json()
            self.assertEqual(impact["schema"], "agentflow.repeated_scaffold_impact.v1")
            self.assertEqual(impact["status"], "matched")
            self.assertEqual(impact["summary"]["applied_count"], 2)
            self.assertEqual(impact["summary"]["holdout_count"], 1)
            self.assertEqual(impact["candidates"][0]["verdict"], "promote")
            self.assertEqual(impact["candidates"][0]["next_action"], "widen_repeated_scaffold_crunch_canary")
            self.assertFalse(impact["privacy"]["raw_request_bodies_included"])
            self.assertFalse(impact["privacy"]["request_ids_included"])
            self.assertFalse(impact["privacy"]["session_ids_included"])
            self.assertFalse(impact["privacy"]["cache_keys_included"])

            self.assertEqual(activation_response.status_code, 200)
            activation = activation_response.json()
            self.assertEqual(activation["schema"], "agentflow.repeated_scaffold_activation.v1")
            self.assertEqual(activation["summary"]["repeated_scaffold_recommended_count"], 3)
            self.assertEqual(activation["summary"]["applied_count"], 2)
            self.assertEqual(activation["summary"]["holdout_count"], 1)
            self.assertEqual(activation["summary"]["feedback_sent_count"], 3)
            self.assertFalse(activation["privacy"]["raw_request_bodies_included"])
            self.assertFalse(activation["privacy"]["optimization_unit_ids_included"])
            self.assertFalse(activation["privacy"]["feedback_payloads_included"])

            self.assertEqual(rollout_health_response.status_code, 200)
            rollout = rollout_health_response.json()
            self.assertEqual(rollout["schema"], "agentflow.scaffold_rollout_health.v1")
            self.assertEqual(rollout["status"], "canary-active")
            self.assertEqual(rollout["summary"]["last_fetch_status"], "ok")
            self.assertEqual(rollout["summary"]["action_count"], 3)
            self.assertEqual(rollout["summary"]["accepted_action_count"], 2)
            self.assertEqual(rollout["summary"]["active_rule_count"], 2)
            self.assertEqual(rollout["summary"]["applied_count"], 2)
            self.assertEqual(rollout["summary"]["holdout_count"], 1)
            self.assertEqual(rollout["summary"]["safety_stop_count"], 0)
            self.assertFalse(rollout["privacy"]["raw_action_payloads_included"])
            self.assertFalse(rollout["privacy"]["yaml_contents_included"])
            self.assertFalse(rollout["active_policy"]["rule_path_included"])

            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Managed scaffold rollout", dashboard.text)
            self.assertIn("scaffold-rollout-health-tbody", dashboard.text)
            self.assertIn("Repeated-scaffold crunch readiness", dashboard.text)
            self.assertIn("Repeated-scaffold policy-decision activation", dashboard.text)
            self.assertIn("repeated-scaffold-activation-tbody", dashboard.text)
            self.assertIn("repeated-scaffold-activation-groups-tbody", dashboard.text)
            self.assertIn("repeated-scaffold-impact-candidates-tbody", dashboard.text)

            rendered = (
                json.dumps(opportunity, sort_keys=True)
                + json.dumps(impact, sort_keys=True)
                + json.dumps(activation, sort_keys=True)
                + json.dumps(rollout, sort_keys=True)
                + dashboard.text
            )
            for forbidden in (
                "raw secret one",
                "raw secret two",
                "raw-request-id-must-not-leak",
                "raw-cache-key-must-not-leak",
                "raw-session-must-not-leak",
                "raw response must not leak",
                "reviewed-provider-scaffold-second",
                "managed-dashboard-rule-must-not-leak",
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            if old_canary_policy is None:
                os.environ.pop("AGENTFLOW_SCAFFOLD_CANARY_POLICY", None)
            else:
                os.environ["AGENTFLOW_SCAFFOLD_CANARY_POLICY"] = old_canary_policy
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()
            policy_tmp.cleanup()

    def test_terminal_output_compaction_dashboard_endpoint_is_content_free(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        policy_tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml")
        policy_tmp.write("raw-dashboard-policy-file-secret: must-not-render\n")
        policy_tmp.flush()
        store = Store(tmp.name)
        try:
            from agentflow_proxy import crunch

            terminal_text = "\n".join(
                [
                    "$ pytest tests/test_dashboard_secret.py",
                    "FAILED tests/test_dashboard_secret.py::test_hidden - AssertionError: raw-dashboard-terminal-secret",
                    "Traceback (most recent call last):",
                    '  File "/workspace/private/tests/test_dashboard_secret.py", line 9, in test_hidden',
                    "2026-06-12T10:00:00Z ERROR pid=1234 dashboard secret failed",
                ]
                * 80
            )
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "raw dashboard prompt secret must not leak"},
                            {
                                "type": "tool_result",
                                "tool_use_id": "raw-dashboard-tool-use-id",
                                "content": [{"type": "text", "text": terminal_text}],
                            }
                        ],
                    }
                ]
            }
            for idx, text_chars in enumerate((48000, 48200)):
                store.log_call(
                    id=f"dashboard-terminal-compaction-{idx}",
                    created_at=f"2026-06-12T10:0{idx}:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=text_chars // 4,
                    output_tokens_est=100,
                    actual_input_tokens=text_chars // 4,
                    actual_output_tokens=100,
                    cost_est_usd=0.04,
                    cost_baseline_usd=0.04,
                    crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
                    routing_json=stable_json({
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "text_chars": text_chars,
                        "has_tools": True,
                        "request_id": "raw-dashboard-request-id",
                    }),
                    cache_json=stable_json({
                        "status": "skipped",
                        "reason": "streaming",
                        "policy_source": "local-default",
                        "cache_key": "raw-dashboard-cache-key",
                    }),
                    error=None,
                    request_json=stable_json(body),
                    response_json=stable_json({"text": "raw dashboard response must not leak"}),
                    session_id="raw-dashboard-session-id",
                    category="tool-result",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    thinking_output_tokens=0,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    requested_model_family="sonnet",
                    routed_model_family="sonnet",
                )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            with patch.object(crunch, "CRUNCH_RULES_PATH", policy_tmp.name):
                response = client.get("/agentflow/stats/terminal-output-compaction?opportunity_limit=10&impact_limit=10")
                dashboard_response = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(dashboard_response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_readiness.v1")
            self.assertTrue(payload["read_only"])
            self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
            self.assertEqual(payload["summary"]["opportunity_candidate_count"], 1)
            self.assertIn(payload["state"], {"disabled", "ready"})
            self.assertFalse(payload["policy"]["rule_file"]["rule_path_included"])
            self.assertFalse(payload["policy"]["rule_file"]["policy_file_contents_included"])
            self.assertIn("Terminal-output compaction readiness", dashboard_response.text)
            self.assertIn("terminal-compaction-summary-tbody", dashboard_response.text)
            self.assertIn("fetch('/agentflow/stats/terminal-output-compaction?opportunity_limit=1000&impact_limit=500')", dashboard_response.text)
            self.assertFalse(payload["privacy"]["raw_terminal_lines_included"])
            self.assertFalse(payload["privacy"]["raw_terminal_text_included"])
            self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["raw_responses_included"])
            self.assertFalse(payload["privacy"]["tool_payloads_included"])
            self.assertFalse(payload["privacy"]["file_paths_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertFalse(payload["privacy"]["session_ids_included"])
            self.assertFalse(payload["privacy"]["cache_keys_included"])
            self.assertFalse(payload["privacy"]["policy_file_contents_included"])
            rendered = json.dumps(payload, sort_keys=True)
            for forbidden in (
                "raw-dashboard-terminal-secret",
                "raw dashboard prompt secret",
                "raw dashboard response",
                "raw-dashboard-tool-use-id",
                "raw-dashboard-request-id",
                "raw-dashboard-session-id",
                "raw-dashboard-cache-key",
                "raw-dashboard-policy-file-secret",
                "tests/test_dashboard_secret.py",
                "/workspace/private",
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            store.conn.close()
            tmp.close()
            policy_tmp.close()

    def test_terminal_output_compaction_activation_endpoint_is_read_only_and_content_free(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        policy_tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml")
        policy_tmp.write("raw-activation-policy-file-secret: must-not-render\n")
        policy_tmp.flush()
        store = Store(tmp.name)
        try:
            from agentflow_proxy import crunch
            from agentflow_proxy.terminal_compaction_feedback import FEEDBACK_SCHEMA, SOURCE_SURFACE

            policy = {
                "enabled": True,
                "policy_source": "managed-recommended",
                "rule_id": "raw activation rule secret / local-path",
                "candidate_id": "raw activation candidate secret",
                "canary": {
                    "enabled": True,
                    "fraction": 0.25,
                    "holdout_fraction": 0.75,
                    "unit": "request_fingerprint",
                    "salt": "raw activation salt secret",
                },
                "safety_stop": {"enabled": True, "min_outcome_samples": 1, "window": 50},
                "rules": [
                    {
                        "enabled": True,
                        "policy_source": "managed-recommended",
                        "rule_id": "raw activation rule secret / local-path",
                        "candidate_id": "raw activation candidate secret",
                        "action_id": "raw activation action secret",
                        "action": {"type": "compact_terminal_output"},
                        "canary": {
                            "enabled": True,
                            "fraction": 0.25,
                            "holdout_fraction": 0.75,
                            "unit": "request_fingerprint",
                            "salt_configured": True,
                        },
                        "safety_stop": {"enabled": True, "min_outcome_samples": 1, "window": 50},
                    }
                ],
            }

            def log_activation_call(call_id, *, status, cohort, reason, changed, tokens_saved=0, planned_tokens=1200):
                terminal_meta = {
                    "schema": "agentflow.terminal_output_compaction_decision.v1",
                    "enabled": True,
                    "status": status,
                    "reason": reason,
                    "changed": changed,
                    "applied": changed,
                    "policy_source": "managed-recommended",
                    "rule_id": "raw activation rule secret / local-path",
                    "candidate_id": "raw activation candidate secret",
                    "action_id": "raw activation action secret",
                    "category": "tool-result",
                    "canary": {
                        "schema": "agentflow.terminal_output_compaction_canary_decision.v1",
                        "enabled": True,
                        "selected": changed,
                        "status": status,
                        "cohort": cohort,
                    },
                    "planned_saved_tokens": planned_tokens,
                    "tokens_saved_est": tokens_saved,
                    "compaction_cost_usd": 0.0,
                    "raw_terminal_text_included": False,
                    "raw_request_body_included": False,
                    "raw_tool_ids_included": False,
                    "raw_session_ids_included": False,
                }
                store.log_call(
                    id=call_id,
                    created_at="2026-06-12T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=10_000,
                    output_tokens_est=100,
                    actual_input_tokens=10_000,
                    actual_output_tokens=100,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.034,
                    crunch_json=stable_json({"changed": changed, "terminal_output_compaction": terminal_meta}),
                    routing_json=stable_json({
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "text_chars": 40_000,
                        "request_id": "raw-activation-request-id",
                    }),
                    cache_json=stable_json({"status": "skipped", "cache_key": "raw-activation-cache-key"}),
                    error=None,
                    request_json=stable_json({"messages": [{"content": "raw activation terminal secret"}]}),
                    response_json=stable_json({"text": "raw activation response secret"}),
                    session_id="raw-activation-session-id",
                    category="tool-result",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    thinking_output_tokens=0,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    requested_model_family="sonnet",
                    routed_model_family="sonnet",
                )

            log_activation_call(
                "activation-applied",
                status="applied",
                cohort="canary_applied",
                reason="terminal-output-compaction-applied",
                changed=True,
                tokens_saved=900,
            )
            log_activation_call(
                "activation-holdout",
                status="holdout",
                cohort="canary_holdout",
                reason="canary_holdout",
                changed=False,
            )
            log_activation_call(
                "activation-safety",
                status="safety_stop",
                cohort="safety_stop",
                reason="safety-stop-error-rate",
                changed=False,
            )
            store.enqueue_managed_outcome_feedback(
                id="raw-activation-queue-id",
                created_at="2026-06-12T10:05:00+00:00",
                updated_at="2026-06-12T10:05:00+00:00",
                source_surface=SOURCE_SURFACE,
                endpoint="/v1/feedback",
                optimization_unit_id=0,
                status="queued",
                payload_json=stable_json({
                    "event_type": "canary-applied",
                    "metadata": {
                        "schema": FEEDBACK_SCHEMA,
                        "action_snapshots": [
                            {
                                "candidate_id": "raw activation candidate secret",
                                "rule_id": "raw activation rule secret / local-path",
                                "lifecycle_status": "safety-stop",
                                "actual_cohort_counts": {"canary_applied": 1, "canary_holdout": 1},
                                "reason_codes": ["safety-stop-error-rate"],
                                "net_savings_usd": 0.004,
                                "projected_saved_tokens": 2400,
                            }
                        ],
                    },
                }),
            )
            for suffix, status in (("retry", "retryable-error"), ("dropped", "dropped-after-limit")):
                store.enqueue_managed_outcome_feedback(
                    id=f"raw-activation-queue-{suffix}",
                    created_at=f"2026-06-12T10:0{6 if suffix == 'retry' else 7}:00+00:00",
                    updated_at=f"2026-06-12T10:0{6 if suffix == 'retry' else 7}:00+00:00",
                    source_surface=SOURCE_SURFACE,
                    endpoint="/v1/feedback",
                    optimization_unit_id=0,
                    status=status,
                    last_error="raw activation feedback failure must stay hidden",
                    last_status_code=503,
                    payload_json=stable_json({
                        "event_type": "rollback" if suffix == "retry" else "rejected",
                        "metadata": {
                            "schema": FEEDBACK_SCHEMA,
                            "action_snapshots": [
                                {
                                    "candidate_id": "raw activation candidate secret",
                                    "rule_id": "raw activation rule secret / local-path",
                                    "lifecycle_status": "rollback" if suffix == "retry" else "rejected",
                                    "reason_codes": [
                                        "rollback-error-rate",
                                        "raw activation terminal secret",
                                        "raw-activation-request-id",
                                    ],
                                    "raw_provider_body": "raw activation provider body must stay hidden",
                                    "request_id": "raw-activation-request-id",
                                    "session_id": "raw-activation-session-id",
                                    "cache_key": "raw-activation-cache-key",
                                    "file_path": "/workspace/private/activation-secret.log",
                                    "tenant_id": "raw-activation-tenant-id",
                                }
                            ],
                        },
                    }),
                )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            with (
                patch.object(crunch, "TERMINAL_OUTPUT_COMPACTION_POLICY", policy),
                patch.object(crunch, "CRUNCH_RULES_PATH", policy_tmp.name),
            ):
                response = client.get("/agentflow/stats/terminal-output-compaction-activation?limit=20")
                dashboard_response = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(dashboard_response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_activation.v1")
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["status"], "rollback-ready")
            self.assertGreaterEqual(payload["summary"]["active_rule_count"], 1)
            self.assertEqual(payload["summary"]["applied_count"], 1)
            self.assertEqual(payload["summary"]["holdout_count"], 1)
            self.assertGreaterEqual(payload["summary"]["safety_stop_count"], 1)
            self.assertTrue(payload["summary"]["rollback_action_ready"])
            self.assertTrue(payload["summary"]["latest_safety_stop_reason"])
            self.assertGreater(payload["summary"]["managed_lifecycle_feedback_rows"], 0)
            self.assertEqual(payload["summary"]["managed_lifecycle_feedback_retryable_error"], 1)
            self.assertEqual(payload["summary"]["managed_lifecycle_feedback_dropped"], 1)
            self.assertFalse(payload["lifecycle_feedback"]["payload_json_included"])
            self.assertIn("Terminal-output compaction activation", dashboard_response.text)
            self.assertIn("terminal-compaction-activation-tbody", dashboard_response.text)
            self.assertIn("fetch('/agentflow/stats/terminal-output-compaction-activation?opportunity_limit=1000&impact_limit=500')", dashboard_response.text)
            self.assertFalse(payload["privacy"]["raw_terminal_lines_included"])
            self.assertFalse(payload["privacy"]["raw_terminal_text_included"])
            self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
            self.assertFalse(payload["privacy"]["tool_payloads_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertFalse(payload["privacy"]["session_ids_included"])
            self.assertFalse(payload["privacy"]["cache_keys_included"])
            self.assertFalse(payload["privacy"]["policy_file_contents_included"])
            rendered = json.dumps(payload, sort_keys=True)
            for forbidden in (
                "raw activation terminal secret",
                "raw activation response secret",
                "raw activation rule secret",
                "raw activation candidate secret",
                "raw activation action secret",
                "raw activation request-id",
                "raw-activation-request-id",
                "raw-activation-session-id",
                "raw-activation-cache-key",
                "raw-activation-policy-file-secret",
                "raw activation feedback failure",
                "raw activation provider body",
                "raw-activation-tenant-id",
                "/workspace/private/activation-secret.log",
                policy_tmp.name,
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            store.conn.close()
            tmp.close()
            policy_tmp.close()

    def test_managed_openai_activation_dashboard_uses_metadata_only_sources(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        draft_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        old_draft_dir = os.environ.get("AGENTFLOW_POLICY_DRAFT_DIR")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        os.environ["AGENTFLOW_POLICY_DRAFT_DIR"] = draft_tmp.name
        store = Store(tmp.name)
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "fetch-review",
                ok=True,
                details={
                    "source": "cli",
                    "status_code": 200,
                    "provenance_status": "verified",
                    "provenance_managed_bundle": True,
                    "openai_optimization_review": {
                        "status": "present",
                        "selected_action_count": 2,
                        "suppressed_action_count": 1,
                        "omitted_action_count": 1,
                        "local_capability_gap_count": 0,
                    },
                    "request_id": "raw fetch request id must stay hidden",
                    "raw_prompt": "raw fetch prompt must stay hidden",
                },
            )
            log_policy_event(
                "openai-optimization-draft-dry-run",
                ok=True,
                details={
                    "source": "cli",
                    "draft": "openai-activation-fixture",
                    "openai_rows_considered": 12,
                    "applied_if_enabled_total": 3,
                    "suppressed_total": 1,
                    "raw_response": "raw dry-run response must stay hidden",
                },
            )
            draft_dir = Path(draft_tmp.name) / "openai-activation-fixture"
            draft_dir.mkdir()
            (draft_dir / "draft.json").write_text(
                json.dumps(
                    {
                        "schema": "agentflow.policy_draft.v1",
                        "draft_id": "openai-activation-fixture",
                        "created_at": "2026-06-11T20:10:00+00:00",
                        "changed": True,
                        "changed_sections": ["routing", "cache"],
                        "change_count": 2,
                        "metadata": {
                            "openai_optimization_review": {
                                "schema": "agentflow.openai_optimization_review_draft_metadata.v1",
                                "source": "openai_optimization_review_bundle",
                                "selected_action_count": 2,
                                "suppressed_action_count": 1,
                                "omitted_action_count": 1,
                                "staged_action_count": 2,
                                "staged_policy_sections": ["routing", "cache"],
                                "counts_by_family": {
                                    "routing": {"selected": 1, "suppressed": 1, "omitted": 0},
                                    "cache": {"selected": 1, "suppressed": 0, "omitted": 1},
                                },
                                "conflict_summary": {
                                    "conflict_count": 1,
                                    "raw_prompt": "raw conflict prompt must stay hidden",
                                },
                                "selected_actions": [
                                    {
                                        "action_family": "routing",
                                        "target_candidate_id": "candidate-routing",
                                        "raw_request": "raw selected action request must stay hidden",
                                    },
                                    {
                                        "action_family": "cache",
                                        "target_candidate_id": "candidate-cache",
                                        "cache_key": "cache key must stay hidden",
                                    },
                                ],
                                "suppressed_actions": [
                                    {
                                        "action_family": "old_context_summarization",
                                        "prompt": "raw suppressed prompt must stay hidden",
                                    }
                                ],
                                "omitted_actions": [
                                    {
                                        "action_family": "cache",
                                        "request_id": "raw omitted request id must stay hidden",
                                    }
                                ],
                            }
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            store.enqueue_managed_outcome_feedback(
                id="openai-activation-queued",
                created_at="2026-06-11T20:15:00+00:00",
                updated_at="2026-06-11T20:15:00+00:00",
                source_surface="openai_optimization_lifecycle",
                endpoint="/v1/policy-events",
                optimization_unit_id=44,
                payload_json=json.dumps({
                    "schema": "agentflow.openai_optimization_draft_dry_run_lifecycle_feedback.v1",
                    "raw_prompt": "raw queued feedback prompt must stay hidden",
                    "provider_body": "raw provider body must stay hidden",
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T00:00:00+00:00",
            )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://api.openai.com",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            response = client.get("/agentflow/stats/managed-openai-activation")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.managed_openai_activation.v1")
            self.assertEqual(payload["status"], "feedback-due")
            self.assertEqual(payload["summary"]["selected_action_count"], 2)
            self.assertEqual(payload["summary"]["suppressed_action_count"], 1)
            self.assertEqual(payload["summary"]["omitted_action_count"], 1)
            self.assertEqual(payload["summary"]["staged_draft_count"], 1)
            self.assertEqual(payload["summary"]["openai_lifecycle_feedback_due"], 1)
            self.assertEqual(payload["bundle_health"]["provenance_status"], "verified")
            self.assertEqual(payload["bundle_health"]["supported_local_action_families"], ["routing", "old_context_summarization", "cache"])
            families = {row["family"]: row for row in payload["bundle_health"]["counts_by_family"]}
            self.assertEqual(families["routing"]["selected"], 1)
            self.assertEqual(families["cache"]["omitted"], 1)
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertFalse(payload["privacy"]["cache_keys_included"])
            self.assertFalse(payload["privacy"]["payload_json_included"])
            self.assertIn("Managed OpenAI activation", dashboard.text)
            self.assertIn("managed-openai-activation-tbody", dashboard.text)

            rendered = json.dumps(payload, sort_keys=True) + dashboard.text
            for forbidden in (
                "raw fetch request id must stay hidden",
                "raw fetch prompt must stay hidden",
                "raw dry-run response must stay hidden",
                "raw conflict prompt must stay hidden",
                "raw selected action request must stay hidden",
                "cache key must stay hidden",
                "raw suppressed prompt must stay hidden",
                "raw omitted request id must stay hidden",
                "raw queued feedback prompt must stay hidden",
                "raw provider body must stay hidden",
                '"raw_prompt"',
                '"request_id"',
                '"cache_key"',
                '"provider_body"',
                '"raw_request"',
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            if old_draft_dir is None:
                os.environ.pop("AGENTFLOW_POLICY_DRAFT_DIR", None)
            else:
                os.environ["AGENTFLOW_POLICY_DRAFT_DIR"] = old_draft_dir
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()
            draft_tmp.cleanup()

    def test_openai_old_context_summary_dashboard_api_reports_impact_without_raw_content(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        try:
            def feature(endpoint="responses", source_surface="openai_responses"):
                return {
                    "schema": "agentflow.openai_feature_summary.v1",
                    "provider": "openai",
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "requested_model_family": "gpt-5",
                    "routed_model_family": "gpt-5",
                    "stream": False,
                    "category": "chat",
                    "workflow_phase": "summary",
                    "text_bucket": "32k_128k_chars",
                    "input_token_bucket": "16k_64k_tokens",
                    "has_tools": False,
                    "old_context": {
                        "shape": "responses_input_items",
                        "conversation_item_count": 12,
                        "older_context_item_count": 8,
                        "older_context_text_bucket": "32k_128k_chars",
                        "older_context_token_bucket": "16k_64k_tokens",
                        "raw_payload_included": False,
                    },
                    "raw_payload_included": False,
                }

            def summary_meta(cohort, status, candidate_id):
                return {
                    "schema": "agentflow.openai_old_context_summary.v1",
                    "enabled": True,
                    "status": status,
                    "applied": status == "applied",
                    "changed": status == "applied",
                    "rule_id": "local-openai-old-context-summary",
                    "candidate_id": candidate_id,
                    "summary_model": "gpt-5-mini",
                    "endpoint": "responses",
                    "workflow_phase": "summary",
                    "canary": {"cohort": cohort, "canary_fraction": 0.5, "holdout_fraction": 0.5},
                    "estimated_tokens_saved": 1000 if status == "applied" else 0,
                    "estimated_gross_savings_usd": 0.005 if status == "applied" else 0.0,
                    "summary_cost_est_usd": 0.001 if status == "applied" else 0.0,
                    "estimated_net_savings_usd": 0.004 if status == "applied" else 0.0,
                    "summary_status_code": 200,
                    "reason_codes": [status],
                    "privacy": {
                        "raw_source_included": False,
                        "raw_summary_included": False,
                        "raw_request_body_included": False,
                        "summary_text_included": False,
                        "session_id_included": False,
                    },
                }

            for idx, (cohort, status) in enumerate((
                ("canary_applied", "applied"),
                ("canary_applied", "applied"),
                ("canary_holdout", "holdout"),
            )):
                routing = {
                    "provider": "openai",
                    "requested_model": "gpt-5.4",
                    "routed_model": "gpt-5.4",
                    "text_chars": 64000,
                    "has_tools": False,
                    "category": "chat",
                    "workflow_phase": "summary",
                    "openai_feature_unit": feature(),
                }
                meta = summary_meta(cohort, status, f"secret-content-derived-candidate-{idx}")
                store.log_call(
                    id=f"openai-summary-dashboard-{idx}",
                    created_at=utc_now(),
                    path="/v1/responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=120,
                    input_tokens_est=16000,
                    output_tokens_est=80,
                    actual_input_tokens=16000,
                    actual_output_tokens=80,
                    cost_est_usd=0.01,
                    cost_baseline_usd=0.02,
                    crunch_json=stable_json({"openai_old_context_summarization": meta}),
                    routing_json=stable_json(routing),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
                    error=None,
                    request_json='{"input":"secret raw openai prompt","request_id":"req_secret","file_path":"/tmp/secret.py"}',
                    response_json='{"output_text":"secret generated response"}',
                    session_id="secret-openai-session",
                    category="chat",
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

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://openai.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
            )
            client = TestClient(app)

            with patch.dict(os.environ, {"AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED": "1"}):
                payload = client.get("/agentflow/stats/openai-old-context-summary?limit=10")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(payload.status_code, 200)
            data = payload.json()
            self.assertEqual(data["summary"]["openai_call_count"], 3)
            self.assertEqual(data["summary"]["eligible_count"], 3)
            self.assertEqual(data["quality_gate_summary"]["canary_applied_count"], 2)
            self.assertEqual(data["quality_gate_summary"]["canary_holdout_count"], 1)
            self.assertGreater(data["quality_gate_summary"]["estimated_net_savings_usd"], 0)
            self.assertEqual(data["quality_gates"][0]["verdict"], "promote")
            self.assertTrue(data["measurement_policy"]["summary_provider_configured"])
            self.assertFalse(data["local_policy"]["rule_file"]["rule_path_included"])
            self.assertFalse(data["privacy"]["raw_request_bodies_included"])
            self.assertFalse(data["privacy"]["request_ids_included"])
            rendered = json.dumps(data, sort_keys=True)
            for forbidden in (
                "secret-openai-session",
                "secret raw openai prompt",
                "secret generated response",
                "req_secret",
                "/tmp/secret.py",
                "secret-content-derived-candidate",
            ):
                self.assertNotIn(forbidden, rendered)
                self.assertNotIn(forbidden, dashboard.text)
        finally:
            store.conn.close()
            tmp.close()

    def test_policy_workbench_readiness_reports_staged_drafts_events_and_privacy(self):
        from agentflow_proxy import stats as stats_views
        from agentflow_proxy.policy_events import log_policy_event

        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        work_tmp = tempfile.TemporaryDirectory()
        store = Store(tmp.name)
        workspace = Path(work_tmp.name) / "drafts"
        event_log = Path(work_tmp.name) / "policy_events.jsonl"
        draft_dir = workspace / "draft-one"
        draft_dir.mkdir(parents=True)
        (draft_dir / "draft.json").write_text(
            json.dumps({
                "schema": "agentflow.policy_draft.v1",
                "draft_id": "draft-one",
                "created_at": "2026-06-10T20:00:00+00:00",
                "requested_section": "cache",
                "changed": True,
                "changed_sections": ["cache"],
                "change_count": 2,
                "workspace": str(draft_dir),
                "bundle_path": str(draft_dir / "policy_bundle.json"),
                "raw_prompt": "raw workbench prompt must not leak",
                "request_id": "workbench-request-id-must-not-leak",
                "session_id": "workbench-session-id-must-not-leak",
                "cache_key": "workbench-cache-key-must-not-leak",
                "sections": [{"section": "cache"}],
                "privacy": {
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "provider_bodies_included": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
            }),
            encoding="utf-8",
        )

        env = {
            "AGENTFLOW_POLICY_DRAFT_DIR": str(workspace),
            "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
        }
        with patch.dict(os.environ, env, clear=False):
            log_policy_event(
                "draft-stage",
                ok=True,
                details={
                    "source": "cli",
                    "draft_id": "draft-one",
                    "changed_sections": ["cache"],
                    "change_count": 2,
                    "raw_prompt": "raw event prompt must not leak",
                    "request_id": "event-request-id-must-not-leak",
                    "session_id": "event-session-id-must-not-leak",
                    "cache_key": "event-cache-key-must-not-leak",
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "draft-validate",
                ok=True,
                details={
                    "source": "cli",
                    "draft": "draft-one",
                    "status": "pass",
                    "can_apply": True,
                    "apply_blocked": False,
                    "changed_sections": ["cache"],
                    "section_verdicts": {"cache": "pass"},
                    "blocker_reason_codes": [],
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "draft-apply",
                ok=True,
                details={
                    "source": "cli",
                    "draft_id": "draft-one",
                    "apply_id": "apply-123",
                    "backup_id": "apply-123",
                    "status": "applied",
                    "changed_sections": ["cache"],
                    "backup_paths": [str(Path(work_tmp.name) / "cache_rules.yaml.bak-apply-123")],
                    "reloaded_modules": ["agentflow_proxy.cache"],
                    "verification_ok": True,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "rollback",
                ok=True,
                details={
                    "source": "cli",
                    "apply_id": "apply-123",
                    "backup_id": "apply-123",
                    "status": "rolled_back",
                    "restored_sections": ["cache"],
                    "reloaded_modules": ["agentflow_proxy.cache"],
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "reload",
                ok=False,
                details={
                    "source": "cli",
                    "error_type": "unsafe_url",
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "exit_code": 2,
                },
            )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
            )
            client = TestClient(app)
            response = client.get("/agentflow/stats/policy-workbench")
            policies = client.get("/agentflow/stats/policies")
            reload_required = asyncio.run(stats_views.stats_policy_workbench_readiness({
                "summary": {"reload_required_sections": ["cache"]},
            }))

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["schema"], "agentflow.policy_workbench_readiness.v1")
        self.assertEqual(payload["staged_drafts"]["count"], 1)
        self.assertEqual(payload["staged_drafts"]["latest"]["draft_id"], "draft-one")
        self.assertFalse(payload["staged_drafts"]["latest"]["workspace_path_included"])
        self.assertEqual(payload["validation"]["latest"]["status"], "pass")
        self.assertTrue(payload["validation"]["can_apply"])
        self.assertEqual(payload["apply"]["latest"]["apply_id"], "apply-123")
        self.assertIn("apply-123", payload["apply"]["last_backup_ids"])
        self.assertEqual(payload["rollback"]["latest"]["restored_sections"], ["cache"])
        self.assertEqual(payload["events"]["latest_failure"]["error_type"], "unsafe_url")
        self.assertFalse(payload["events"]["raw_path_included"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["mutating_dashboard_endpoints"])
        self.assertFalse(payload["privacy"]["absolute_paths_included"])
        self.assertFalse(payload["privacy"]["draft_bundle_contents_included"])
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        rendered = json.dumps(payload, sort_keys=True)
        for sensitive in (
            "raw workbench prompt must not leak",
            "workbench-request-id-must-not-leak",
            "workbench-session-id-must-not-leak",
            "workbench-cache-key-must-not-leak",
            "raw event prompt must not leak",
            "event-request-id-must-not-leak",
            "event-session-id-must-not-leak",
            "event-cache-key-must-not-leak",
        ):
            self.assertNotIn(sensitive, rendered)
        self.assertEqual(policies.status_code, 200)
        self.assertEqual(policies.json()["workbench"]["staged_drafts"]["count"], 1)
        self.assertEqual(reload_required["status"], "reload-required")
        self.assertEqual(reload_required["reload"]["required_sections"], ["cache"])

        store.conn.close()
        tmp.close()
        work_tmp.cleanup()

    def test_optimization_eval_queue_endpoint_and_dashboard_are_metadata_only(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)

        def plan_row(
            candidate_id,
            *,
            action_family="routing",
            optimization_family="phase_routing",
            applied=2,
            holdout=1,
            applied_error_rate=0.0,
            holdout_error_rate=0.0,
            blockers=None,
        ):
            return {
                "schema": "agentflow.optimization_eval_plan_row.v1",
                "candidate_id": candidate_id,
                "optimization_family": optimization_family,
                "action_family": action_family,
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "granularity": "provider_request",
                "workflow_phase": "tool_execution",
                "category": "tool-result",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "candidate_profile": "fixture-profile",
                "projected_savings_usd": 0.0123,
                "sample_count": applied + holdout,
                "current_canary_count": applied,
                "holdout_count": holdout,
                "blocker_reason_codes": blockers or [],
                "recommended_eval_mode": "score-canary-holdout" if applied or holdout else "run-local-shadow-eval",
                "replayability_level": "features_only",
                "evidence": {
                    "canary_evidence": {
                        "applied": {
                            "count": applied,
                            "error_rate": applied_error_rate,
                            "retry_rate": 0.0,
                            "latency_avg_ms": 100,
                            "net_savings_usd": 0.0123,
                            "tool_payload": {"command": "raw eval queue tool payload must stay local"},
                        },
                        "holdout": {
                            "count": holdout,
                            "error_rate": holdout_error_rate,
                            "retry_rate": 0.0,
                            "latency_avg_ms": 110,
                            "content": "raw eval queue holdout content must stay local",
                        },
                    },
                    "api_key": "sk-eval-queue-secret",
                    "cache_key": "eval-queue-cache-key-secret",
                    "content": "raw eval queue content must stay local",
                    "messages": [{"content": "raw eval queue message must stay local"}],
                    "raw_prompt": "raw eval queue prompt must stay local",
                    "raw_request": {"body": "raw eval queue request must stay local"},
                    "raw_response": {"body": "raw eval queue response must stay local"},
                    "request_json": {"body": "raw provider body must stay local"},
                    "request_id": "eval-queue-request-id-secret",
                    "session_id": "eval-queue-session-secret",
                    "tool_payload": {"args": "raw eval queue tool must stay local"},
                    "file_path": "/tmp/eval-queue-secret.py",
                },
                "privacy": {"metadata_only": True},
            }

        fake_plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                plan_row("queue-widen"),
                plan_row("queue-hold", applied_error_rate=0.1),
                plan_row("queue-rollback"),
                plan_row("queue-needs-eval", applied=0, holdout=0),
                plan_row("queue-privacy", applied=0, holdout=0, blockers=["raw-provider-body-present"]),
            ],
        }

        async def fake_build_optimization_eval_plan(store_obj, *, limit=500, min_samples=1):
            self.assertIs(store_obj, store)
            self.assertEqual(min_samples, 1)
            return fake_plan

        try:
            for candidate_id, status in (
                ("queue-widen", "pass"),
                ("queue-hold", "pass"),
                ("queue-rollback", "fail"),
            ):
                store.log_optimization_eval_result(
                    id=f"eval-queue-{candidate_id}",
                    run_id="eval-queue-run",
                    created_at="2026-06-10T04:00:00+00:00",
                    candidate_id=candidate_id,
                    source_surface="anthropic_messages",
                    optimization_family="phase_routing",
                    action_family="routing",
                    status_class=status,
                    reason_codes_json=stable_json([
                        "offline-fixture-passed" if status == "pass" else "output-similarity-below-threshold"
                    ]),
                    score_json=stable_json({"output_similarity": 0.99 if status == "pass" else 0.1}),
                    cost_json=stable_json({"projected_savings_usd": 0.0123}),
                    result_json=stable_json({
                        "candidate_id": candidate_id,
                        "status_class": status,
                        "api_key": "sk-eval-queue-result-secret",
                        "cache_key": "eval-queue-result-cache-key-secret",
                        "content": "raw eval result content must stay local",
                        "prompt": "raw eval result prompt must stay local",
                        "raw_request": {"messages": [{"content": "raw eval result request must stay local"}]},
                        "raw_response": {"content": "raw eval result response body must stay local"},
                        "response": "raw eval result response must stay local",
                        "request_id": "eval-queue-request-secret",
                        "session_id": "eval-queue-result-session-secret",
                        "tool_payload": {"command": "raw eval result tool must stay local"},
                    }),
                )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            with patch("agentflow_proxy.optimization_eval_plan.build_optimization_eval_plan", fake_build_optimization_eval_plan):
                response = client.get("/agentflow/stats/optimization-eval-queue?limit=50")
                dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.optimization_eval_queue.v1")
            by_candidate = {row["candidate_id"]: row for row in payload["candidates"]}
            self.assertEqual(by_candidate["queue-widen"]["verdict"], "widen")
            self.assertEqual(by_candidate["queue-hold"]["verdict"], "hold")
            self.assertEqual(by_candidate["queue-rollback"]["verdict"], "rollback")
            self.assertEqual(by_candidate["queue-needs-eval"]["verdict"], "needs_eval")
            self.assertEqual(by_candidate["queue-privacy"]["queue_status"], "privacy_blocked")
            self.assertEqual(by_candidate["queue-widen"]["workflow_phase"], "tool_execution")
            self.assertEqual(by_candidate["queue-widen"]["category"], "tool-result")
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["raw_provider_bodies_included"])
            self.assertFalse(payload["privacy"]["provider_calls_made"])
            self.assertIn("optimization-eval-candidates-tbody", dashboard.text)

            rendered = json.dumps(payload, sort_keys=True) + dashboard.text
            for forbidden in (
                "raw eval queue prompt must stay local",
                "raw eval queue message must stay local",
                "raw eval queue content must stay local",
                "raw eval queue request must stay local",
                "raw eval queue response must stay local",
                "raw eval queue tool payload must stay local",
                "raw eval queue tool must stay local",
                "raw provider body must stay local",
                "raw eval result content must stay local",
                "raw eval result prompt must stay local",
                "raw eval result response must stay local",
                "raw eval result request must stay local",
                "raw eval result response body must stay local",
                "raw eval result tool must stay local",
                "sk-eval-queue-secret",
                "sk-eval-queue-result-secret",
                "eval-queue-cache-key-secret",
                "eval-queue-result-cache-key-secret",
                "eval-queue-session-secret",
                "eval-queue-result-session-secret",
                "eval-queue-request-secret",
                "eval-queue-request-id-secret",
                "/tmp/eval-queue-secret.py",
                '"api_key"',
                '"cache_key"',
                '"content"',
                '"messages"',
                '"raw_request"',
                '"raw_response"',
                '"request_json"',
                '"raw_prompt"',
                '"prompt"',
                '"response"',
                '"request_id"',
                '"session_id"',
                '"tool_payload"',
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            store.conn.close()
            tmp.close()

    def test_optimization_promotion_funnel_endpoint_and_dashboard_are_metadata_only(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        store = Store(tmp.name)

        def plan_row(
            candidate_id,
            *,
            applied=0,
            holdout=0,
            projected=0.02,
            action_family="routing",
            optimization_family="phase_routing",
            blocker_reason_codes=None,
        ):
            return {
                "schema": "agentflow.optimization_eval_plan_row.v1",
                "candidate_id": candidate_id,
                "optimization_family": optimization_family,
                "action_family": action_family,
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "granularity": "provider_request",
                "workflow_phase": "tool_execution",
                "category": "tool-result",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "projected_savings_usd": projected,
                "sample_count": max(1, applied + holdout),
                "current_canary_count": applied,
                "holdout_count": holdout,
                "blocker_reason_codes": blocker_reason_codes or [],
                "recommended_eval_mode": "score-canary-holdout",
                "replayability_level": "features_only",
                "evidence": {
                    "canary_evidence": {
                        "applied": {"count": applied, "error_rate": 0.0, "net_savings_usd": projected},
                        "holdout": {"count": holdout, "error_rate": 0.0},
                    },
                    "raw_prompt": "raw promotion funnel prompt must stay local",
                    "request_id": "promotion-funnel-request-secret",
                    "cache_key": "promotion-funnel-cache-secret",
                    "file_path": "/tmp/promotion-funnel-secret.py",
                },
                "privacy": {"metadata_only": True},
            }

        fake_plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                plan_row("promotion-widen", applied=2, holdout=1, projected=0.05),
                plan_row("promotion-eval-passed", projected=0.03),
                plan_row("promotion-crunch-pending", action_family="crunch", optimization_family="pattern_crunch", projected=0.02),
                plan_row(
                    "promotion-cache-invalidation",
                    action_family="cache",
                    optimization_family="cache_replay",
                    projected=0.04,
                    blocker_reason_codes=["missing-invalidation-evidence"],
                ),
                plan_row(
                    "promotion-old-context-rollback",
                    action_family="old_context_summarization",
                    optimization_family="old_context_summarization",
                    projected=0.025,
                ),
                plan_row("promotion-safety-stop", projected=0.01),
            ],
        }

        async def fake_build_optimization_eval_plan(store_obj, *, limit=500, min_samples=1):
            self.assertIs(store_obj, store)
            return fake_plan

        def log_canary_call(call_id, candidate_id, *, status, cohort, status_code=200, retry_count=0, baseline=0.03, cost=0.01):
            store.log_call(
                id=call_id,
                created_at="2026-06-10T05:00:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001" if status == "applied" else "claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=status_code,
                latency_ms=100 if cohort == "canary_applied" else 120,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=1000,
                actual_output_tokens=100,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "phase_canary": {
                        "promotion_action_id": f"promotion-action-{candidate_id}",
                        "target_candidate_id": candidate_id,
                        "policy_section": "routing",
                        "policy_source": "managed-recommended",
                        "status": status,
                        "cohort": cohort,
                        "reason": "selected-canary" if status == "applied" else ("selected-holdout" if status == "holdout" else "local-canary-safety-stop"),
                        "canary_fraction": 0.1,
                        "holdout_fraction": 0.1,
                        "safety_stop": {
                            "tripped": status == "safety_stopped",
                            "reason_codes": ["error-rate"] if status == "safety_stopped" else [],
                        },
                    },
                    "raw_request": "raw promotion funnel routing body must stay local",
                }),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw promotion funnel request must stay local"}]}),
                response_json=stable_json({"content": "raw promotion funnel response must stay local"}),
                session_id="promotion-funnel-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
                source_surface="anthropic_messages",
            )

        try:
            from agentflow_proxy.policy_events import log_policy_event

            store.log_optimization_eval_result(
                id="promotion-widen-eval",
                run_id="promotion-funnel-run",
                created_at="2026-06-10T04:00:00+00:00",
                candidate_id="promotion-widen",
                source_surface="anthropic_messages",
                optimization_family="phase_routing",
                action_family="routing",
                status_class="pass",
                reason_codes_json=stable_json(["offline-fixture-passed"]),
                score_json=stable_json({"output_similarity": 0.99}),
                cost_json=stable_json({"projected_savings_usd": 0.05}),
                result_json=stable_json({"raw_response": "raw promotion eval result must stay local"}),
            )
            store.log_optimization_eval_result(
                id="promotion-eval-passed-eval",
                run_id="promotion-funnel-run",
                created_at="2026-06-10T04:05:00+00:00",
                candidate_id="promotion-eval-passed",
                source_surface="anthropic_messages",
                optimization_family="phase_routing",
                action_family="routing",
                status_class="pass",
                reason_codes_json=stable_json(["offline-fixture-passed"]),
                score_json=stable_json({"output_similarity": 0.99}),
                cost_json=stable_json({"projected_savings_usd": 0.03}),
                result_json=stable_json({"request_id": "promotion-eval-result-request-secret"}),
            )
            store.log_optimization_eval_result(
                id="promotion-old-context-rollback-eval",
                run_id="promotion-funnel-run",
                created_at="2026-06-10T04:10:00+00:00",
                candidate_id="promotion-old-context-rollback",
                source_surface="anthropic_messages",
                optimization_family="old_context_summarization",
                action_family="old_context_summarization",
                status_class="fail",
                reason_codes_json=stable_json(["offline-fixture-failed"]),
                score_json=stable_json({"output_similarity": 0.25}),
                cost_json=stable_json({"projected_savings_usd": 0.025}),
                result_json=stable_json({"raw_response": "raw old-context promotion eval result must stay local"}),
            )
            log_canary_call("promotion-crunch-pending-applied", "promotion-crunch-pending", status="applied", cohort="canary_applied")
            log_canary_call("promotion-crunch-pending-holdout", "promotion-crunch-pending", status="holdout", cohort="canary_holdout", baseline=0.03, cost=0.03)
            log_canary_call("promotion-safety-stopped", "promotion-safety-stop", status="safety_stopped", cohort="bypassed_or_disabled", status_code=500, retry_count=1)
            store.enqueue_managed_outcome_feedback(
                id="promotion-crunch-pending-feedback",
                created_at="2026-06-10T05:15:00+00:00",
                updated_at="2026-06-10T05:15:00+00:00",
                source_surface="optimization_promotion_lifecycle",
                endpoint="/v1/policy-events",
                optimization_unit_id=0,
                payload_json=json.dumps({
                    "event_type": "impact",
                    "occurred_at": "2026-06-10T05:15:00+00:00",
                    "metadata": {
                        "schema": "agentflow.optimization_promotion_lifecycle_feedback.v1",
                        "command": "optimization-promotion-impact",
                        "candidate_ids": ["promotion-crunch-pending"],
                        "action_ids": ["promotion-action-promotion-crunch-pending"],
                        "policy_section_counts": {"crunch": 1},
                        "actual_canary_applied_count": 1,
                        "actual_canary_holdout_count": 1,
                        "raw_prompt": "raw queued lifecycle prompt must stay local",
                        "privacy": {
                            "metadata_only": True,
                            "raw_prompts_included": False,
                            "request_ids_included": False,
                            "file_paths_included": False,
                        },
                    },
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T00:00:00+00:00",
            )
            log_policy_event(
                "optimization-promotion-canary-impact",
                ok=True,
                details={
                    "candidate_ids": ["promotion-crunch-pending"],
                    "action_ids": ["promotion-action-promotion-crunch-pending"],
                    "actual_canary_applied_count": 1,
                    "actual_canary_holdout_count": 1,
                    "path": "/tmp/raw-promotion-action.json",
                    "request_id": "policy-event-request-secret",
                    "cache_key": "policy-event-cache-secret",
                    "raw_prompt": "raw policy event prompt must stay local",
                    "reason": "canary-observed",
                },
            )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            with patch("agentflow_proxy.optimization_eval_plan.build_optimization_eval_plan", fake_build_optimization_eval_plan):
                response = client.get("/agentflow/stats/optimization-promotion-funnel?limit=50")
                dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.optimization_promotion_funnel.v1")
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertFalse(payload["privacy"]["cache_keys_included"])
            by_candidate = {row["candidate_id"]: row for row in payload["candidates"]}
            self.assertEqual(by_candidate["promotion-widen"]["primary_state"], "widening-eligible")
            self.assertEqual(by_candidate["promotion-eval-passed"]["primary_state"], "eval-passed")
            self.assertEqual(by_candidate["promotion-crunch-pending"]["primary_state"], "canary-active")
            self.assertEqual(by_candidate["promotion-safety-stop"]["primary_state"], "safety-stopped")
            self.assertEqual(by_candidate["promotion-crunch-pending"]["canary"]["applied_count"], 1)
            self.assertEqual(by_candidate["promotion-crunch-pending"]["canary"]["holdout_count"], 1)
            self.assertGreater(by_candidate["promotion-crunch-pending"]["observed_savings_usd"], 0)
            self.assertEqual(by_candidate["promotion-widen"]["policy_section"], "routing")
            self.assertEqual(by_candidate["promotion-widen"]["executor_readiness"]["status"], "widening-eligible")
            self.assertEqual(by_candidate["promotion-crunch-pending"]["policy_section"], "crunch")
            self.assertEqual(by_candidate["promotion-crunch-pending"]["executor_readiness"]["status"], "pending-lifecycle-feedback")
            self.assertEqual(by_candidate["promotion-cache-invalidation"]["policy_section"], "cache")
            self.assertEqual(by_candidate["promotion-cache-invalidation"]["executor_readiness"]["status"], "missing-invalidation-evidence")
            self.assertEqual(by_candidate["promotion-old-context-rollback"]["policy_section"], "old_context_summarization")
            self.assertEqual(by_candidate["promotion-old-context-rollback"]["executor_readiness"]["status"], "rollback-recommended")
            self.assertEqual(by_candidate["promotion-crunch-pending"]["next_command_kind"], "promotion-impact")
            self.assertEqual(by_candidate["promotion-widen"]["next_command_kind"], "promotion-actions")
            self.assertEqual(by_candidate["promotion-eval-passed"]["next_command_kind"], "promotion-canaries-apply --dry-run")
            self.assertIn(
                "routing:widening-eligible",
                {row["value"] for row in payload["executor_readiness_by_policy_section"]},
            )
            self.assertIn(
                "cache:missing-invalidation-evidence",
                {row["value"] for row in payload["executor_readiness_by_policy_section"]},
            )
            self.assertEqual(payload["summary"]["canary_applied_count"], 1)
            self.assertEqual(payload["summary"]["canary_holdout_count"], 1)
            self.assertEqual(payload["summary"]["pending_lifecycle_feedback_count"], 1)
            self.assertIn("Optimization promotion canary impact", dashboard.text)
            self.assertIn("optimization-promotion-funnel-candidates-tbody", dashboard.text)
            self.assertIn("Readiness", dashboard.text)
            self.assertIn("Next command", dashboard.text)

            rendered = json.dumps(payload, sort_keys=True) + dashboard.text
            for forbidden in (
                "raw promotion funnel prompt must stay local",
                "promotion-funnel-request-secret",
                "promotion-funnel-cache-secret",
                "/tmp/promotion-funnel-secret.py",
                "raw promotion funnel routing body must stay local",
                "raw promotion funnel request must stay local",
                "raw promotion funnel response must stay local",
                "promotion-funnel-session-secret",
                "raw promotion eval result must stay local",
                "promotion-eval-result-request-secret",
                "raw old-context promotion eval result must stay local",
                "raw queued lifecycle prompt must stay local",
                "/tmp/raw-promotion-action.json",
                "policy-event-request-secret",
                "policy-event-cache-secret",
                "raw policy event prompt must stay local",
                '"raw_prompt"',
                '"request_id"',
                '"cache_key"',
                '"file_path"',
                '"session_id"',
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()

    def test_claude_routing_promotion_funnel_endpoint_and_dashboard_are_metadata_only(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)

        def log_call(call_id, *, suffix, status="applied", cohort="canary_applied", cost=0.01, baseline=0.03):
            store.log_call(
                id=call_id,
                created_at=f"2026-06-11T05:00:0{suffix}+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001" if status == "applied" else "claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=1200,
                output_tokens_est=100,
                actual_input_tokens=1200,
                actual_output_tokens=100,
                cost_est_usd=cost,
                cost_baseline_usd=baseline,
                routing_json=stable_json({
                    "workflow_phase": "tool-execution",
                    "phase_canary": {
                        "target_candidate_id": "claude-public-routing-candidate",
                        "target_model": "claude-haiku-4-5-20251001",
                        "original_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "stream": True,
                        "status": status,
                        "cohort": cohort,
                        "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                        "canary_fraction": 0.2,
                        "holdout_fraction": 0.1,
                    },
                    "raw_prompt": "raw claude canary prompt must stay local",
                    "request_id": "claude-canary-request-secret",
                    "session_id": "claude-canary-session-secret",
                    "file_path": "/tmp/claude-canary-secret.py",
                    "tool_payload": {"secret": "claude-canary-tool-secret"},
                }),
                crunch_json=stable_json({"changed": False}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw claude request body must stay local"}]}),
                response_json=stable_json({"content": "raw claude response body must stay local"}),
                session_id="claude-routing-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
                source_surface="anthropic_messages",
            )

        try:
            store.log_call(
                id="claude-eligible-unsampled",
                created_at="2026-06-11T04:59:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=90,
                input_tokens_est=900,
                output_tokens_est=80,
                actual_input_tokens=900,
                actual_output_tokens=80,
                cost_est_usd=0.02,
                cost_baseline_usd=0.02,
                routing_json=stable_json({
                    "workflow_phase": "tool-execution",
                    "category": "tool-result",
                    "candidate_target_model": "claude-haiku-4-5-20251001",
                    "claude_routing_promotion": {
                        "eligible": True,
                        "stage": "eligible",
                        "reason": "eligible-not-sampled",
                    },
                    "raw_response": "raw eligible response must stay local",
                    "authorization": "Bearer claude-eligible-secret",
                }),
                crunch_json=stable_json({"changed": False}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=stable_json({"messages": [{"content": "raw eligible request must stay local"}]}),
                response_json=stable_json({"content": "raw eligible response body must stay local"}),
                session_id="claude-eligible-session-secret",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
                source_surface="anthropic_messages",
            )
            for idx in range(3):
                store.log_routing_experiment(
                    id=f"claude-shadow-{idx}",
                    call_id=f"claude-shadow-call-secret-{idx}",
                    created_at=f"2026-06-11T05:01:0{idx}+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    stream=1,
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    routing_reason="sampled-claude-shadow-routing",
                    input_tokens_est=1000,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=140,
                    shadow_latency_ms=90,
                    primary_output_chars=80,
                    shadow_output_chars=78,
                    primary_output_sha256=f"primary-claude-{idx}",
                    shadow_output_sha256=f"shadow-claude-{idx}",
                    output_similarity=0.97,
                    passed_threshold=1,
                    primary_cost_est_usd=0.03,
                    shadow_cost_est_usd=0.01,
                    budget_limit_usd=1.0,
                    budget_spent_before_usd=0.10,
                    budget_remaining_before_usd=0.90,
                    budget_spent_after_usd=0.11,
                    error=None,
                    routing_json=stable_json({"workflow_phase": "tool-execution", "request_id": "claude-shadow-request-secret"}),
                    experiment_json=stable_json({
                        "workflow_phase": "tool-execution",
                        "raw_prompt": "raw claude shadow prompt must stay local",
                        "provider_body": {"messages": [{"content": "raw claude provider body must stay local"}]},
                        "tool_payload": {"secret": "claude-shadow-tool-secret"},
                    }),
                    primary_response_json=stable_json({"content": "raw claude primary response must stay local"}),
                    shadow_response_json=stable_json({"content": "raw claude shadow response must stay local"}),
                )
            log_call("claude-canary-applied-1", suffix=1)
            log_call("claude-canary-applied-2", suffix=2)
            log_call("claude-canary-holdout-1", suffix=3, status="holdout", cohort="canary_holdout", cost=0.03, baseline=0.03)

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            response = client.get("/agentflow/stats/claude-routing-promotion-funnel?limit=50")
            yield_response = client.get("/agentflow/stats/post-fix-shadow-yield?window_hours=0&limit=50")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.claude_routing_promotion_funnel.v1")
            self.assertTrue(payload["read_only"])
            self.assertFalse(payload["provider_calls_made"])
            self.assertFalse(payload["managed_server_calls_made"])
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["raw_provider_bodies_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertGreaterEqual(payload["summary"]["eligible_unsampled_count"], 1)
            self.assertGreaterEqual(payload["summary"]["compared_count"], 3)
            self.assertGreaterEqual(payload["summary"]["canary_applied_count"], 2)
            self.assertGreaterEqual(payload["summary"]["holdout_count"], 1)
            self.assertGreaterEqual(payload["summary"]["widened_count"], 1)
            self.assertGreaterEqual(payload["summary"]["promoted_count"], 1)
            self.assertTrue(any(row["eligible_unsampled_count"] >= 1 for row in payload["candidates"]))
            self.assertTrue(any(row["compared_count"] >= 3 for row in payload["candidates"]))
            self.assertTrue(any(row["canary_widened_count"] >= 1 for row in payload["candidates"]))
            self.assertEqual(yield_response.status_code, 200)
            yield_payload = yield_response.json()
            self.assertEqual(yield_payload["schema"], "agentflow.post_fix_shadow_yield.v1")
            self.assertGreaterEqual(yield_payload["summary"]["sample_count"], 3)
            self.assertGreaterEqual(yield_payload["summary"]["compared_count"], 3)
            self.assertFalse(yield_payload["privacy"]["raw_prompts_included"])
            self.assertFalse(yield_payload["privacy"]["request_ids_included"])
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Claude routing promotion funnel", dashboard.text)
            self.assertIn("/agentflow/stats/claude-routing-promotion-funnel", dashboard.text)
            self.assertIn("claude-routing-funnel-candidates-tbody", dashboard.text)
            self.assertIn("post-fix-shadow-yield-tbody", dashboard.text)

            rendered = json.dumps(payload, sort_keys=True) + json.dumps(yield_payload, sort_keys=True) + dashboard.text
            for forbidden in (
                "raw claude canary prompt must stay local",
                "claude-canary-request-secret",
                "claude-canary-session-secret",
                "/tmp/claude-canary-secret.py",
                "claude-canary-tool-secret",
                "raw claude request body must stay local",
                "raw claude response body must stay local",
                "claude-routing-session-secret",
                "raw eligible request must stay local",
                "raw eligible response must stay local",
                "Bearer claude-eligible-secret",
                "claude-shadow-call-secret",
                "claude-shadow-request-secret",
                "raw claude shadow prompt must stay local",
                "raw claude provider body must stay local",
                "claude-shadow-tool-secret",
                "raw claude primary response must stay local",
                "raw claude shadow response must stay local",
                '"request_id"',
                '"session_id"',
                '"file_path"',
                '"tool_payload"',
                '"provider_body"',
                '"raw_prompt"',
            ):
                self.assertNotIn(forbidden, rendered)
        finally:
            store.conn.close()
            tmp.close()

    def test_local_pattern_coverage_endpoint_summarizes_family_readiness(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        try:
            families = [
                "terminal_logs",
                "tool_results",
                "diffs",
                "generated_artifacts",
                "tabular_data",
                "cacheability",
            ]
            modules = [
                {
                    "schema": "agentflow.local_pattern_module_outcome.v1",
                    "family": family,
                    "version": "test",
                    "enabled": True,
                    "supports_local_crunch": family in {"tool_results", "diffs", "generated_artifacts"},
                    "local_crunch_enabled": family in {"tool_results", "diffs", "generated_artifacts"},
                    "status": "applied" if family == "tool_results" else "skipped",
                    "reason": "fixture-local-crunch-applied" if family == "tool_results" else "feature-only-no-local-crunch",
                    "detected": True,
                    "features_emitted": True,
                    "changed": family == "tool_results",
                    "saved_chars": 64 if family == "tool_results" else 0,
                    "tokens_saved_est": 16 if family == "tool_results" else 0,
                    "privacy_guard": {"safe": True, "violation_count": 0, "blocked_keys": [], "raw_values_logged": False},
                    "feature_summary": {
                        "family": family,
                        "feature_schema": f"agentflow.{family}.fixture_features.v1",
                        "raw_content_included": False,
                    },
                }
                for family in families
            ]
            crunch_json = {
                "changed": True,
                "saved_chars": 64,
                "tokens_saved_est": 16,
                "pattern_modules": {
                    "schema": "agentflow.local_pattern_modules.v1",
                    "registered_count": len(families),
                    "enabled_count": len(families),
                    "detected_count": len(families),
                    "features_emitted_count": len(families),
                    "applied_count": 1,
                    "bypass_count": 0,
                    "modules": modules,
                    "server_features": {
                        "schema": "agentflow.local_pattern_module_features.v1",
                        "module_feature_count": len(families),
                        "features": [
                            {
                                "family": family,
                                "version": "test",
                                "feature_schema": f"agentflow.{family}.fixture_features.v1",
                                "features": {
                                    "schema": f"agentflow.{family}.fixture_features.v1",
                                    "module_family": family,
                                    "detected": True,
                                    "privacy": {"metadata_only": True, "raw_content_included": False},
                                },
                            }
                            for family in families
                        ],
                        "privacy": {
                            "metadata_only": True,
                            "raw_content_included": False,
                            "raw_provider_body_included": False,
                            "raw_tool_payload_included": False,
                        },
                    },
                    "raw_content_included": False,
                },
            }
            routing_json = {
                "category": "tool-result",
                "managed_pattern_features": {
                    "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                    "present": True,
                    "pattern_hash_count": 3,
                    "pattern_hashes": [
                        "sha256:" + "1" * 64,
                        "sha256:" + "2" * 64,
                        "sha256:" + "3" * 64,
                    ],
                    "pattern_hash": "sha256:" + "1" * 64,
                    "crunch_pattern_hash": "sha256:" + "2" * 64,
                    "cache_pattern_hash": "sha256:" + "3" * 64,
                    "hash_basis": "normalized-structure-and-size-buckets",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "workflow_phase": "tool-result",
                    "local_pattern_module_families": families,
                    "local_pattern_module_count": len(families),
                    "pattern_types": families,
                    "raw_pattern_strings_included": False,
                },
            }
            store.log_call(
                id="pattern-coverage-fixture",
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
                cost_est_usd=0.001,
                cost_baseline_usd=0.001,
                crunch_json=stable_json(crunch_json),
                routing_json=stable_json(routing_json),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-pattern-coverage",
                category="tool-result",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )
            app = create_dashboard_app(
                store_obj=store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
            )
            client = TestClient(app)

            response = client.get("/agentflow/stats/local-pattern-coverage?limit=50")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["schema"], "agentflow.local_pattern_coverage.v1")
            self.assertFalse(data["privacy"]["raw_prompts_included"])
            self.assertFalse(data["privacy"]["raw_tool_payloads_included"])
            by_family = {row["family"]: row for row in data["families"]}
            for family in families:
                self.assertIn(family, by_family)
                self.assertEqual(by_family[family]["detected_call_count"], 1)
                self.assertEqual(by_family[family]["fingerprint_count"], 3)
                self.assertFalse(by_family[family]["raw_content_included"])
                self.assertIn("recommendation-fetch-disabled", by_family[family]["managed_eligibility"]["reasons"])
            self.assertEqual(by_family["tool_results"]["applied_count"], 1)
            self.assertIn("local_crunch", by_family["tool_results"]["action_families_seen"])
        finally:
            store.conn.close()
            tmp.close()

    def test_rollout_action_readiness_endpoint_summarizes_metadata_only(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        store = Store(tmp.name)
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "rollout-actions-review",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/raw-action-payload.json",
                    "config_dir": "/tmp/local-yaml-config",
                    "action_count": 2,
                    "planned_action_count": 2,
                    "changed_action_count": 1,
                    "provenance_status": "verified",
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "rollout-actions-dry-run",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/raw-action-payload.json",
                    "config_dir": "/tmp/local-yaml-config",
                    "db_path": tmp.name,
                    "dry_run": True,
                    "action_count": 2,
                    "affected_metadata_row_count": 9,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "pattern-canary-safety-stop",
                ok=True,
                details={
                    "reason": "local-canary-safety-stop",
                    "policy_section": "crunch",
                    "rule_id": "rule-id-not-rendered",
                    "pattern_hash": "sha256:" + ("b" * 64),
                    "sample_count": 6,
                    "raw_payload_included": False,
                },
            )
            log_policy_event(
                "rollout-actions-impact",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/dry-run-report.json",
                    "db_path": tmp.name,
                    "action_count": 2,
                    "projected_affected_metadata_row_count": 9,
                    "actual_matched_metadata_row_count": 4,
                    "actual_matched_provider_call_count": 3,
                    "actual_matched_codex_turn_count": 1,
                    "actual_canary_applied_count": 2,
                    "actual_canary_holdout_count": 1,
                    "actual_bypassed_or_disabled_count": 1,
                    "actual_tokens_saved_est": 700,
                    "actual_estimated_cost_savings_usd": 0.007,
                    "actions_without_post_apply_matches": 0,
                    "exit_code": 0,
                },
            )
            store.enqueue_managed_outcome_feedback(
                id="rollout-feedback-queued",
                created_at="2026-06-09T03:40:00+00:00",
                updated_at="2026-06-09T03:40:00+00:00",
                source_surface="rollout_action_lifecycle",
                endpoint="/v1/policy-events",
                optimization_unit_id=0,
                payload_json=json.dumps({
                    "event_type": "dry-run",
                    "occurred_at": "2026-06-09T03:40:00+00:00",
                    "bundle_hash": "sha256:" + ("c" * 64),
                    "action_ids": ["must-not-render-action-id"],
                    "metadata": {
                        "schema": "agentflow.rollout_action_lifecycle_metadata.v1",
                        "command": "rollout-actions-dry-run",
                        "local_result_status": "ok",
                        "dry_run": True,
                        "read_only": True,
                        "action_count": 2,
                        "planned_action_count": 2,
                        "changed_action_count": 1,
                        "action_type_counts": {"widen": 1, "rollback": 1},
                        "policy_section_counts": {"crunch": 2},
                        "local_status_counts": {"planned": 2},
                        "affected_metadata_row_count": 9,
                        "affected_provider_call_count": 5,
                        "affected_codex_turn_count": 4,
                        "projected_additional_applied_count": 3,
                        "projected_local_bypass_or_disable_count": 2,
                        "historical_tokens_saved_est": 1200,
                        "historical_estimated_cost_savings_usd": 0.0123,
                        "safety_stop_reason_counts": {"local-canary-safety-stop": 1},
                        "candidate_ids": ["candidate-id-not-rendered"],
                        "rule_ids": ["rule-id-not-rendered"],
                        "pattern_hashes": ["sha256:" + ("b" * 64)],
                        "raw_prompt": "raw prompt must stay hidden",
                        "yaml_contents": "local YAML must stay hidden",
                        "privacy": {
                            "metadata_only": True,
                            "raw_prompts_included": False,
                            "raw_messages_included": False,
                            "raw_responses_included": False,
                            "raw_transcripts_included": False,
                            "raw_params_included": False,
                            "tool_payloads_included": False,
                            "request_ids_included": False,
                            "local_session_ids_included": False,
                            "file_paths_included": False,
                            "yaml_contents_included": False,
                        },
                    },
                    "raw_request": "raw request body must stay hidden",
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T09:00:00+00:00",
            )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            response = client.get("/agentflow/stats/rollout-actions/readiness")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.rollout_actions_readiness.v1")
            self.assertEqual(payload["summary"]["pending_lifecycle_feedback_count"], 1)
            self.assertEqual(payload["summary"]["due_lifecycle_feedback_count"], 1)
            self.assertEqual(payload["summary"]["affected_metadata_row_count"], 9)
            self.assertEqual(payload["dry_run_impact"]["projected_additional_applied_count"], 3)
            self.assertEqual(payload["dry_run_impact"]["projected_local_bypass_or_disable_count"], 2)
            self.assertEqual(payload["latest_impact"]["stage"], "impact")
            self.assertEqual(payload["post_apply_impact"]["actual_matched_metadata_row_count"], 4)
            self.assertEqual(payload["post_apply_impact"]["actual_canary_applied_count"], 2)
            self.assertEqual({row["value"]: row["count"] for row in payload["action_type_counts"]}, {"widen": 1, "rollback": 1})
            self.assertTrue(payload["safety_stop"]["active"])
            self.assertFalse(payload["privacy"]["raw_action_payloads_included"])
            self.assertFalse(payload["privacy"]["yaml_contents_included"])
            self.assertIn("rollout-readiness-tbody", dashboard.text)
            self.assertIn("rollout-action-counts-tbody", dashboard.text)

            rendered = json.dumps(payload) + dashboard.text
            self.assertNotIn("raw prompt must stay hidden", rendered)
            self.assertNotIn("raw request body must stay hidden", rendered)
            self.assertNotIn("local YAML must stay hidden", rendered)
            self.assertNotIn("must-not-render-action-id", rendered)
            self.assertNotIn("candidate-id-not-rendered", rendered)
            self.assertNotIn("rule-id-not-rendered", rendered)
            self.assertNotIn("/tmp/raw-action-payload.json", rendered)
            self.assertNotIn("/tmp/dry-run-report.json", rendered)
            self.assertNotIn("/tmp/local-yaml-config", rendered)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()

    def test_safety_stats_warn_and_redact_unsafe_configuration(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        env = {
            "AGENTFLOW_LOG_BODIES": "1",
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
            "AGENTFLOW_RECOMMENDATION_SERVER_URL": "https://user:supersecret@managed.test/v1/recommendation?api_key=managedsecret&mode=dev",
            "AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL": "https://managed.test/v1/policy-bundle-recommendation?token=policysecret",
            "AGENTFLOW_MANAGED_API_KEY": "",
            "AGENTFLOW_POLICY_EVENTS": "0",
        }
        try:
            with patch.dict(os.environ, env, clear=False):
                app = create_dashboard_app(
                    store_obj=lambda: store,
                    default_db=tmp.name,
                    upstream="https://anthropic.test",
                    limiter_status=lambda: [],
                    limiter_config={},
                    proxy_host="0.0.0.0",
                    dashboard_host="0.0.0.0",
                    full_stats_ttl_s=0,
                )
                client = TestClient(app)
                response = client.get("/agentflow/stats/safety")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            warning_codes = {row["code"] for row in payload["warnings"]}
            self.assertIn("proxy-bind-non-loopback", warning_codes)
            self.assertIn("body-logging-enabled", warning_codes)
            self.assertIn("managed-recommendation-unauthenticated", warning_codes)
            self.assertIn("managed-policy-fetch-unauthenticated", warning_codes)
            self.assertIn("policy-events-disabled", warning_codes)
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
            self.assertFalse(payload["checks"]["managed"]["api_key_value_included"])
            redacted = str(payload)
            self.assertIn("[redacted]", redacted)
            self.assertNotIn("supersecret", redacted)
            self.assertNotIn("managedsecret", redacted)
            self.assertNotIn("policysecret", redacted)
            self.assertNotIn("user:supersecret", redacted)
        finally:
            store.conn.close()
            tmp.close()

    def test_safety_stats_warn_on_stuck_managed_feedback_queue_without_payloads(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        try:
            store.enqueue_managed_outcome_feedback(
                id="queue-due",
                created_at="2000-01-01T09:00:00+00:00",
                updated_at="2000-01-01T09:00:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/77/outcome",
                optimization_unit_id=77,
                payload_json=json.dumps({
                    "prompt": "must not appear in dashboard",
                    "raw_response": "provider body must stay local",
                    "params": {"secret": "raw params must stay local"},
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T09:00:00+00:00",
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-retry",
                created_at="2000-01-01T09:05:00+00:00",
                updated_at="2000-01-01T09:05:00+00:00",
                source_surface="anthropic_messages",
                endpoint="/v1/optimization-units/88/outcome",
                optimization_unit_id=88,
                payload_json=json.dumps({"messages": ["raw prompt text"]}),
                status="retryable-error",
                attempts=2,
                next_attempt_at="2000-01-01T09:05:00+00:00",
                last_error="ConnectError: managed feedback down",
                last_status_code=503,
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-dropped",
                created_at="2000-01-01T09:10:00+00:00",
                updated_at="2000-01-01T09:10:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/99/outcome",
                optimization_unit_id=99,
                payload_json=json.dumps({"raw_request": "dropped raw request"}),
                status="dropped-after-limit",
                attempts=3,
                next_attempt_at="2000-01-01T09:10:00+00:00",
                last_error="HTTP 500: raw failure body",
                last_status_code=500,
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-sent",
                created_at="2000-01-01T08:55:00+00:00",
                updated_at="2000-01-01T09:15:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/66/outcome",
                optimization_unit_id=66,
                payload_json=json.dumps({"content": "sent raw content"}),
                status="sent",
                attempts=1,
                next_attempt_at="2000-01-01T08:55:00+00:00",
                sent_at="2000-01-01T09:15:00+00:00",
                last_status_code=200,
            )

            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_MANAGED_API_KEY": "test-key",
                },
                clear=False,
            ):
                app = create_dashboard_app(
                    store_obj=lambda: store,
                    default_db=tmp.name,
                    upstream="https://anthropic.test",
                    limiter_status=lambda: [],
                    limiter_config={},
                    full_stats_ttl_s=0,
                )
                client = TestClient(app)
                response = client.get("/agentflow/stats/safety")
                dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            queue = payload["checks"]["managed"]["feedback_queue"]
            warning_codes = {row["code"] for row in payload["warnings"]}
            self.assertIn("managed-feedback-due-queue", warning_codes)
            self.assertIn("managed-feedback-retryable-errors", warning_codes)
            self.assertIn("managed-feedback-dropped-after-limit", warning_codes)
            self.assertEqual(queue["summary"]["queued"], 1)
            self.assertEqual(queue["summary"]["due"], 2)
            self.assertEqual(queue["summary"]["retryable_error"], 1)
            self.assertEqual(queue["summary"]["dropped_after_limit"], 1)
            self.assertEqual(queue["last_successful_flush"]["optimization_unit_id"], 66)
            self.assertFalse(queue["due_samples"][0]["payload_included"])
            self.assertFalse(queue["privacy"]["payload_json_included"])
            self.assertFalse(payload["privacy"]["managed_feedback_payload_json_included"])
            self.assertIn("safety-managed-feedback-tbody", dashboard.text)
            self.assertIn("managed-feedback-queue-tbody", dashboard.text)
            rendered = json.dumps(payload) + dashboard.text
            self.assertNotIn("must not appear in dashboard", rendered)
            self.assertNotIn("provider body must stay local", rendered)
            self.assertNotIn("raw params must stay local", rendered)
            self.assertNotIn("raw prompt text", rendered)
            self.assertNotIn("dropped raw request", rendered)
            self.assertNotIn("sent raw content", rendered)
        finally:
            store.conn.close()
            tmp.close()

    def test_full_stats_endpoint_coalesces_concurrent_requests(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        old_stats_full = dashboard_app.stats_views.stats_full
        call_count = {"value": 0}

        async def fake_stats_full(store_obj):
            call_count["value"] += 1
            await asyncio.sleep(0.05)
            return {
                "summary": {"total_calls": call_count["value"]},
                "generated_by": call_count["value"],
            }

        dashboard_app.stats_views.stats_full = fake_stats_full
        try:
            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
                full_stats_ttl_s=60,
            )

            async def exercise():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    responses = await asyncio.gather(
                        client.get("/agentflow/stats/full"),
                        client.get("/agentflow/stats/full"),
                        client.get("/agentflow/stats/full"),
                    )
                    warm_start = time.perf_counter()
                    cached = await client.get("/agentflow/stats/full")
                    warm_seconds = time.perf_counter() - warm_start
                    return responses, cached, warm_seconds

            responses, cached, warm_seconds = asyncio.run(exercise())

            self.assertEqual([response.status_code for response in responses], [200, 200, 200])
            self.assertEqual(cached.status_code, 200)
            self.assertEqual(call_count["value"], 1)
            self.assertEqual([response.json()["generated_by"] for response in responses], [1, 1, 1])
            self.assertEqual(cached.json()["generated_by"], 1)
            self.assertLess(warm_seconds, 0.2)
        finally:
            dashboard_app.stats_views.stats_full = old_stats_full
            store.conn.close()
            tmp.close()


if __name__ == "__main__":
    unittest.main()
