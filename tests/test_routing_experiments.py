import asyncio
import importlib
import json
import os
import yaml
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.managed_egress import assert_managed_egress_safe
from tokenclaw.recommendations import queue_policy_event_feedback
from tokenclaw.store import Store, stable_json, utc_now
from tokenclaw import stats
import tokenclaw.routing_experiments as experiments


class RoutingExperimentPolicyTest(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_ROUTING_EXPERIMENTS",
        "TOKENCLAW_ROUTING_EXPERIMENTS_STRICT",
        "TOKENCLAW_ROUTING_EXPERIMENTS_ENABLED",
        "TOKENCLAW_ROUTING_EXPERIMENT_SAMPLE_RATE",
        "TOKENCLAW_ROUTING_EXPERIMENT_DAILY_BUDGET_USD",
        "TOKENCLAW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD",
        "TOKENCLAW_RECOMMENDATIONS_ENABLED",
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_POLICY_DECISIONS_ENABLED",
        "TOKENCLAW_POLICY_DECISION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
        "TOKENCLAW_MANAGED_ROUTING",
        "TOKENCLAW_MANAGED_API_KEY",
        "TOKENCLAW_CONFIG_DIR",
        "TOKENCLAW_POLICY_CONFIG_DIR",
        "HOME",
    )

    def _enable_managed_backing(self):
        """Mark routing as server-backed so local-default shadow experiments run.

        Under the backed-or-off rule the bundled default policy mints no local
        canaries unless an opted-in server backs routing, so the default-policy
        evidence-collection tests must simulate a healthy managed backend.
        """
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"

    def _disable_managed_backing(self):
        for key in (
            "TOKENCLAW_RECOMMENDATIONS_ENABLED",
            "TOKENCLAW_RECOMMENDATION_ENABLED",
            "TOKENCLAW_POLICY_DECISIONS_ENABLED",
            "TOKENCLAW_POLICY_DECISION_ENABLED",
            "TOKENCLAW_RECOMMENDATION_SERVER_URL",
            "TOKENCLAW_MANAGED",
            "TOKENCLAW_MANAGED_MODE",
            "TOKENCLAW_MANAGED_ROUTING",
            "TOKENCLAW_MANAGED_API_KEY",
        ):
            os.environ.pop(key, None)

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name
        os.chdir(self.home.name)
        # Default fixture represents a server-backed install; tests that exercise
        # the unbacked/off path opt out via _disable_managed_backing().
        self._enable_managed_backing()
        importlib.reload(experiments)

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(experiments)

    def test_managed_route_policy_prefers_client_pathway_and_blocks_disallowed_target(self):
        config_dir = Path(self.home.name) / "config"
        config_dir.mkdir()
        (config_dir / "routing_experiments.yaml").write_text(
            """
enabled: true
preferred_pathways:
  - requested_model: gpt-5.5
    routed_model: gpt-5-mini
    provider: openai
    source_surface: openai_responses
blocklist:
  - requested_model: gpt-5.5
    routed_model: gpt-5-nano
""",
            encoding="utf-8",
        )
        importlib.reload(experiments)

        override = experiments.routing_pathway_policy_decision(
            provider="openai",
            requested_model="gpt-5.5",
            current_model="gpt-5.5",
            target_model="gpt-5.4",
            source_surface="openai_responses",
        )
        blocked = experiments.routing_pathway_policy_decision(
            provider="openai",
            requested_model="gpt-5.5",
            current_model="gpt-5.5",
            target_model="gpt-5-nano",
            source_surface="openai_responses",
        )

        self.assertEqual(override["decision"], "preferred-override")
        self.assertEqual(override["target_model"], "gpt-5-mini")
        self.assertTrue(override["allowed"])

        (config_dir / "routing_experiments.yaml").write_text(
            """
enabled: true
blocklist:
  - requested_model: gpt-5.5
    routed_model: gpt-5-nano
""",
            encoding="utf-8",
        )
        importlib.reload(experiments)
        blocked = experiments.routing_pathway_policy_decision(
            provider="openai",
            requested_model="gpt-5.5",
            current_model="gpt-5.5",
            target_model="gpt-5-nano",
            source_surface="openai_responses",
        )
        self.assertEqual(blocked["decision"], "blocked")
        self.assertFalse(blocked["allowed"])

    def test_distills_legacy_candidate_config_to_thin_policy(self):
        thin = experiments.distill_thin_routing_policy({
            "enabled": True,
            "mode": "shadow_candidate_pass_through",
            "model_pairs": [
                {"requested_model": "claude-opus-5-0", "routed_model": "claude-sonnet-4-6"},
            ],
            "routing_candidates": [
                {
                    "candidate_id": "preferred-chat",
                    "requested_model": "gpt-5.5",
                    "routed_model": "gpt-5-mini",
                    "provider": "openai",
                    "source_surface": "openai_responses",
                }
            ],
            "blocklist": ["gpt-5-nano"],
        })

        self.assertNotIn("model_pairs", thin)
        self.assertNotIn("routing_candidates", thin)
        preferred_pairs = {
            (item["requested_model"], item["routed_model"])
            for item in thin["preferred_pathways"]
        }
        self.assertIn(("gpt-5.5", "gpt-5-mini"), preferred_pairs)
        self.assertEqual(thin["blocklist"], [{"model": "gpt-5-nano"}])

    def test_openai_shadow_samples_only_server_managed_target(self):
        meta = experiments.routing_experiment_decision(
            {"model": "gpt-5.4", "input": "hello"},
            {
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "chat",
                "workflow_phase": "summary",
                "text_chars": 1200,
                "managed_recommendation": {
                    "selected_for_shadow_evaluation": True,
                    "shadow_model": "gpt-5.4-mini",
                    "policy_id": "managed-shadow-gpt54-mini",
                    "policy_source": "managed-recommended",
                },
            },
            stream=False,
            provider="openai",
            source_surface="openai_responses",
            random_value=lambda: 0.99,
        )

        self.assertTrue(meta["sampled"])
        self.assertTrue(meta["sampled_by_canary"])
        self.assertEqual(meta["trigger"], "managed-policy-routing-canary")
        self.assertEqual(meta["candidate_selector"], "forced-managed-policy-routing-canary")
        self.assertEqual(meta["candidate_policy_shape"], "managed-shadow")
        self.assertEqual(meta["candidate_id"], "managed-shadow-gpt54-mini")
        self.assertEqual(meta["policy_source"], "managed-recommended")
        self.assertEqual(meta["shadow_model"], "gpt-5.4-mini")
        self.assertEqual(meta["routed_model"], "gpt-5.4-mini")
        self.assertEqual(meta["primary_model"], "gpt-5.4")

    def test_local_policy_does_not_originate_anthropic_canary(self):
        # Backed or off: even with a managed server configured (the setUp default),
        # the local policy must not originate an anthropic canary. Anthropic canaries
        # are server-directed only (executed via _managed_shadow_experiment_decision);
        # there is no local fallback, regardless of request size.
        meta = experiments.routing_experiment_decision(
            {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]},
            {
                "requested_model": "claude-opus-4-8",
                "routed_model": "claude-opus-4-8",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "text_chars": 200000,
            },
            stream=True,
            provider="anthropic",
            source_surface="anthropic_messages",
            random_value=lambda: 0.0,
        )

        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "no-backed-routing")
        self.assertEqual(meta["backing_reason"], "local-policy-without-managed-backing")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_default_policy_drops_dead_local_anthropic_origination(self):
        # The bundled default policy no longer carries anthropic canary scaffolding:
        # anthropic origination is dead code (the gate above blocks it), so the
        # candidates, model pairs, fallback routes, streaming-shadow surface, and the
        # anthropic text-size eligibility caps (incl. the 128k tool-result cap) were
        # removed. Only the server-forced OpenAI/codex path keeps using this policy.
        policy = experiments._default_experiment_policy()

        self.assertEqual(policy["providers"], ["openai"])
        self.assertNotIn("anthropic_messages", policy["source_surfaces"])
        self.assertEqual(policy["streaming_shadow_source_surfaces"], [])
        self.assertEqual(policy["eligibility_overrides"], [])
        self.assertFalse(
            any("claude" in c.get("requested_model", "") for c in policy["routing_candidates"])
        )
        self.assertFalse(
            any("claude" in p.get("requested_model", "") for p in policy["model_pairs"])
        )
        self.assertFalse(
            any("claude" in r.get("requested_model", "") for r in policy["fallback_routes"])
        )

    def test_response_comparison_produces_similarity_and_hashes(self):
        primary = {"content": [{"type": "text", "text": "summarize the build output"}]}
        shadow = {"content": [{"type": "text", "text": "summarize the build output"}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 26)
        self.assertEqual(result["shadow_output_chars"], 26)
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])
        self.assertEqual(result["output_similarity"], 1.0)
        self.assertTrue(result["passed_threshold"])

    def test_response_comparison_extracts_openai_output_text_without_raw_storage(self):
        primary = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok from primary"}],
                }
            ]
        }
        shadow = {"choices": [{"message": {"content": "ok from primary"}}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 15)
        self.assertEqual(result["shadow_output_chars"], 15)
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])
        self.assertEqual(result["output_similarity"], 1.0)

    def test_response_comparison_scores_matching_tool_calls_on_tool_only_turns(self):
        # Tool-execution turns answer with tool_use blocks and no prose. Text-only
        # similarity was ~0 even when both models chose the same action, so these
        # canaries could never pass quality and routing could never promote. The
        # comparison must score the tool calls.
        primary = {
            "content": [
                {"type": "tool_use", "id": "a", "name": "read_file", "input": {"path": "main.py"}}
            ]
        }
        shadow = {
            "content": [
                {"type": "tool_use", "id": "b", "name": "read_file", "input": {"path": "main.py"}}
            ]
        }

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 0)
        self.assertEqual(result["shadow_output_chars"], 0)
        self.assertEqual(result["primary_tool_call_count"], 1)
        self.assertEqual(result["shadow_tool_call_count"], 1)
        self.assertEqual(result["output_similarity"], 1.0)
        self.assertTrue(result["passed_threshold"])
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])

    def test_response_comparison_distinguishes_different_tool_choices(self):
        primary = {"content": [{"type": "tool_use", "name": "read_file", "input": {"path": "a.py"}}]}
        shadow = {"content": [{"type": "tool_use", "name": "run_tests", "input": {"suite": "all"}}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertLess(result["output_similarity"], 1.0)
        self.assertNotEqual(result["primary_output_sha256"], result["shadow_output_sha256"])

    def test_report_explains_unqualified_traffic_with_decision_reasons(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="call-streaming-skip",
                    created_at=utc_now(),
                    path="/v1/responses",
                    provider="openai",
                    source_surface="openai_responses",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-codex",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=10,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "streaming",
                            "mode": "shadow_candidate_pass_through",
                            "provider": "openai",
                            "source_surface": "openai_responses",
                        }
                    }),
                    category="short-completion",
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        self.assertEqual(report["policy"]["profile_id"], "first-safe-openai-codex-claude-shadow-pass-through-v1")
        self.assertEqual(report["summary"]["sample_count"], 0)
        self.assertEqual(report["policy"]["mode"], experiments.ROUTING_EXPERIMENT_MODE)
        self.assertEqual(report["decision_reasons"][0]["provider"], "openai")
        self.assertEqual(report["decision_reasons"][0]["source_surface"], "openai_responses")
        self.assertEqual(report["decision_reasons"][0]["reason"], "streaming")
        self.assertEqual(report["decision_reasons"][0]["count"], 1)

    def test_report_projects_claude_streaming_requests_newly_eligible_under_scoped_caps(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
min_text_chars: 0
max_text_chars: 8000
providers:
  - anthropic
  - openai
source_surfaces:
  - anthropic_messages
  - openai_responses
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
  - requested_model: gpt-5.4
    routed_model: gpt-5.4-mini
categories:
  - chat
  - tool-result
eligibility_overrides:
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    sample_rate: 0.25
    daily_budget_usd: 2.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="claude-large-stream-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=100,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "request-too-large",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "text_chars": 90000,
                        },
                        "raw_prompt": "raw projection secret",
                        "file_path": "/tmp/projection-secret.py",
                    }),
                    category="tool-result",
                )
                store.log_call(
                    id="openai-large-secret",
                    created_at=utc_now(),
                    path="/v1/responses",
                    provider="openai",
                    source_surface="openai_responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=100,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 9000,
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "request-too-large",
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "category": "chat",
                            "text_chars": 9000,
                        },
                    }),
                    category="chat",
                )
                report = manual.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        projection = report["eligibility_projection"]
        [claude_row] = projection["claude_streaming"]
        self.assertEqual(claude_row["category"], "tool-result")
        self.assertEqual(claude_row["observed_call_count"], 1)
        self.assertEqual(claude_row["global_cap_eligible_count"], 0)
        self.assertEqual(claude_row["effective_cap_eligible_count"], 1)
        self.assertEqual(claude_row["newly_eligible_call_count"], 1)
        self.assertEqual(claude_row["effective_max_text_chars_scope"], "category")
        self.assertEqual(claude_row["effective_sample_rate"], 0.25)
        openai_row = next(row for row in projection["rows"] if row["provider"] == "openai")
        self.assertEqual(openai_row["effective_max_text_chars_scope"], "global")
        self.assertEqual(openai_row["newly_eligible_call_count"], 0)
        rendered = stable_json(report)
        self.assertNotIn("raw projection secret", rendered)
        self.assertNotIn("/tmp/projection-secret.py", rendered)
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(projection["privacy"]["file_paths_included"])

    def test_report_explains_claude_shadow_yield_for_streaming_tool_result_and_blocked_row(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
min_text_chars: 0
max_text_chars: 8000
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - chat
  - tool-result
eligibility_overrides:
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    sample_rate: 1.0
    daily_budget_usd: 10.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="claude-stream-tool-result-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=20,
                    input_tokens_est=1000,
                    output_tokens_est=20,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.03,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "schema": "tokenclaw.routing_experiment_decision.v1",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "status": "selected",
                            "sampled": True,
                            "reason": "streaming-shadow-sampled",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-haiku-4-5-20251001",
                            "shadow_model": "claude-haiku-4-5-20251001",
                            "category": "tool-result",
                            "stream": True,
                            "text_chars": 90000,
                        },
                        "raw_prompt": "raw claude yield secret",
                        "file_path": "/tmp/claude-yield-secret.py",
                    }),
                    category="tool-result",
                )
                store.log_call(
                    id="claude-stream-chat-blocked-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=20,
                    input_tokens_est=1000,
                    output_tokens_est=20,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.03,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "schema": "tokenclaw.routing_experiment_decision.v1",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "status": "skipped",
                            "sampled": False,
                            "reason": "request-too-large",
                            "skip_diagnostic": "global-max-text-chars-exceeded",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-haiku-4-5-20251001",
                            "shadow_model": "claude-haiku-4-5-20251001",
                            "category": "chat",
                            "stream": True,
                            "text_chars": 90000,
                        },
                    }),
                    category="chat",
                )
                store.log_routing_experiment(
                    id="claude-sampled-stream-tool-result",
                    call_id="claude-stream-tool-result-secret",
                    created_at=utc_now(),
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    routing_reason="streaming-shadow-sampled",
                    input_tokens_est=1000,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=100,
                    shadow_latency_ms=80,
                    primary_output_chars=20,
                    shadow_output_chars=18,
                    primary_output_sha256="primary",
                    shadow_output_sha256="shadow",
                    output_similarity=0.94,
                    passed_threshold=1,
                    primary_cost_est_usd=0.03,
                    shadow_cost_est_usd=0.005,
                    routing_json=stable_json({}),
                    experiment_json=stable_json({"mode": "shadow_candidate_pass_through"}),
                )
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        yield_report = report["claude_shadow_yield"]
        self.assertEqual(yield_report["schema"], "tokenclaw.claude_shadow_routing_yield.v1")
        self.assertEqual(yield_report["summary"]["observed_call_count"], 2)
        self.assertEqual(yield_report["summary"]["selected_count"], 1)
        self.assertEqual(yield_report["summary"]["skipped_count"], 1)
        self.assertEqual(yield_report["summary"]["sampled_count"], 1)
        self.assertEqual(yield_report["summary"]["compared_count"], 1)
        self.assertIn({"reason": "request-too-large", "count": 1}, yield_report["skipped_reason_counts"])
        self.assertIn({"reason": "request-too-large", "count": 1}, yield_report["cap_block_reason_counts"])
        tool_row = next(row for row in yield_report["observed"] if row["category"] == "tool-result")
        self.assertTrue(tool_row["stream"])
        self.assertEqual(tool_row["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(tool_row["effective_max_text_chars"], 128000)
        self.assertEqual(tool_row["effective_max_text_chars_scope"], "category")
        self.assertEqual(tool_row["selected_count"], 1)
        chat_row = next(row for row in yield_report["observed"] if row["category"] == "chat")
        self.assertEqual(chat_row["ineligible_count"], 1)
        self.assertEqual(chat_row["decision_reasons"], [{"reason": "request-too-large", "count": 1}])
        self.assertEqual(yield_report["projection"]["projected_samples_current_sample_rate"], 1.0)
        self.assertEqual(yield_report["projection"]["projected_samples_sample_rate_100pct"], 1.0)
        rendered_yield = stable_json(yield_report)
        self.assertNotIn("raw claude yield secret", rendered_yield)
        self.assertNotIn("/tmp/claude-yield-secret.py", rendered_yield)
        self.assertFalse(yield_report["privacy"]["filesystem_paths_included"])
        self.assertFalse(yield_report["privacy"]["request_ids_included"])

    def test_report_counts_codex_app_event_decision_reasons_without_sample_rows(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_codex_app_event(
                    id="codex-start-skip",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="turn-secret",
                    thread_id="thread-secret",
                    message_chars=100,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=42,
                    session_id="session-secret",
                    routing_json=stable_json({
                        "status": "skipped",
                        "reason": "codex-turn-start-model-field-absent",
                        "routing_experiment": {
                            "schema": "tokenclaw.routing_experiment_decision.v1",
                            "provider": "openai",
                            "source_surface": "codex_turn",
                            "status": "skipped",
                            "reason": "missing-requested-model",
                            "sampled": False,
                        },
                    }),
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        self.assertEqual(report["summary"]["sample_count"], 0)
        self.assertEqual(report["summary"]["decision_count"], 1)
        self.assertEqual(report["summary"]["decision_status_counts"], {"skipped": 1})
        self.assertEqual(report["decision_surfaces"], [
            {"provider": "openai", "source_surface": "codex_turn", "status": "skipped", "count": 1}
        ])
        self.assertEqual(report["decision_reasons"][0]["provider"], "openai")
        self.assertEqual(report["decision_reasons"][0]["source_surface"], "codex_turn")
        self.assertEqual(report["decision_reasons"][0]["reason"], "missing-requested-model")
        rendered = stable_json(report)
        self.assertNotIn("turn-secret", rendered)
        self.assertNotIn("thread-secret", rendered)
        self.assertNotIn("session-secret", rendered)

    def test_store_attaches_shadow_coverage_metadata_to_uncovered_proxy_calls(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="uncovered-openai-call",
                    created_at=utc_now(),
                    path="/v1/chat/completions",
                    provider="openai",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=10,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "chat",
                        "raw_prompt": "raw secret prompt should not enter experiment metadata",
                    }),
                    category="chat",
                )
                row = store.conn.execute(
                    "select provider, source_surface, routing_json from calls where id = ?",
                    ("uncovered-openai-call",),
                ).fetchone()
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        routing = json.loads(row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertEqual(row["source_surface"], "openai_chat")
        self.assertEqual(experiment["schema"], "tokenclaw.routing_experiment_decision.v1")
        self.assertEqual(experiment["status"], "skipped")
        self.assertEqual(experiment["reason"], "routing-experiment-decision-missing")
        self.assertEqual(experiment["coverage_class"], "blocked")
        self.assertEqual(experiment["source_surface"], "openai_chat")
        rendered_experiment = stable_json(experiment)
        self.assertNotIn("raw secret prompt", rendered_experiment)
        self.assertFalse(experiment["privacy"]["raw_prompts_included"])
        self.assertEqual(report["summary"]["routing_experiment_denominator_count"], 1)
        self.assertEqual(report["summary"]["routing_experiment_metadata_coverage_rate"], 1.0)
        self.assertEqual(report["summary"]["eligible_count"], 1)
        self.assertEqual(report["summary"]["blocked_count"], 1)
        self.assertEqual(report["summary"]["not_sampled_count"], 0)
        self.assertEqual(report["summary"]["out_of_scope_count"], 0)
        self.assertEqual(report["summary"]["metadata_missing_count"], 0)

    def test_report_denominator_includes_codex_turns_without_sample_rows(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_codex_app_event(
                    id="codex-uncovered-turn",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="turn-secret",
                    thread_id="thread-secret",
                    message_chars=50,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=50,
                    session_id="session-secret",
                    routing_json=stable_json({
                        "status": "observed",
                        "raw_prompt": "raw codex secret should not enter experiment metadata",
                    }),
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
                row = store.conn.execute(
                    "select routing_json from codex_app_events where id = ?",
                    ("codex-uncovered-turn",),
                ).fetchone()
            finally:
                store.conn.close()

        routing = json.loads(row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertEqual(experiment["provider"], "openai")
        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["reason"], "routing-experiment-decision-missing")
        self.assertEqual(report["summary"]["routing_experiment_denominator_count"], 1)
        self.assertEqual(report["summary"]["decision_coverage_counts"]["blocked"], 1)
        self.assertEqual(report["decision_surfaces"], [
            {"provider": "openai", "source_surface": "codex_turn", "status": "skipped", "count": 1}
        ])
        rendered = stable_json(report)
        self.assertNotIn("turn-secret", rendered)
        self.assertNotIn("thread-secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("raw codex secret", stable_json(experiment))

    def test_report_groups_anthropic_openai_and_codex_samples_separately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                for idx, (provider, surface, requested, routed, shadow) in enumerate([
                    ("anthropic", "anthropic_messages", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
                    ("openai", "openai_responses", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-mini"),
                    ("openai", "codex_turn", "gpt-5-codex", "gpt-5-mini", "gpt-5-mini"),
                ]):
                    store.log_routing_experiment(
                        id=f"sample-{idx}",
                        call_id=f"call-{idx}",
                        created_at=utc_now(),
                        provider=provider,
                        source_surface=surface,
                        requested_model=requested,
                        routed_model=routed,
                        primary_model=requested,
                        shadow_model=shadow,
                        category="short-completion" if surface != "codex_turn" else "codex-turn",
                        routing_reason="sampled-shadow-candidate-pass-through",
                        input_tokens_est=100,
                        primary_status_code=200,
                        shadow_status_code=200,
                        primary_latency_ms=100,
                        shadow_latency_ms=90,
                        primary_output_chars=2,
                        shadow_output_chars=2,
                        primary_output_sha256=f"primary-{idx}",
                        shadow_output_sha256=f"shadow-{idx}",
                        output_similarity=1.0,
                        passed_threshold=1,
                        primary_cost_est_usd=0.003,
                        shadow_cost_est_usd=0.001,
                        routing_json=stable_json({}),
                        experiment_json=stable_json({"mode": "shadow_candidate_pass_through"}),
                    )
                report = experiments.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        surfaces = {
            (item["provider"], item["source_surface"], item["requested_model"], item["routed_model"])
            for item in report["candidates"]
        }
        self.assertIn(("anthropic", "anthropic_messages", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"), surfaces)
        self.assertIn(("openai", "openai_responses", "gpt-5.4", "gpt-5.4-mini"), surfaces)
        self.assertIn(("openai", "codex_turn", "gpt-5-codex", "gpt-5-mini"), surfaces)

    def test_feedback_features_are_metadata_only(self):
        comparison = {
            "primary_output_chars": 17,
            "shadow_output_chars": 19,
            "primary_output_sha256": "primary-hash",
            "shadow_output_sha256": "shadow-hash",
            "output_similarity": 0.91,
            "passed_threshold": True,
        }

        result = experiments.routing_experiment_feedback_features(
            experiment_id="exp-1",
            experiment_meta={
                "sampled": True,
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
                "mode": "shadow_candidate_pass_through",
                "counterfactual": True,
                "shadow_only": True,
            },
            routing_meta={
                "category": "tool-result",
                "reason": "tool-result processing turn routed to Haiku",
            },
            comparison=comparison,
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )

        self.assertEqual(result["schema"], "tokenclaw.routing_experiment_feedback.v1")
        self.assertEqual(result["mode"], "shadow_candidate_pass_through")
        self.assertTrue(result["counterfactual"])
        self.assertTrue(result["shadow_only"])
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["candidate_bucket"], "tool-result:claude-sonnet-4-6->claude-haiku-4-5-20251001")
        self.assertEqual(result["text_chars_bucket"], "2k-8k")
        self.assertEqual(result["output_similarity"], 0.91)
        self.assertEqual(result["primary_output_sha256"], "primary-hash")
        self.assertNotIn("text", result)
        self.assertEqual(result["reason_codes"], ["passed"])
        self.assertTrue(result["privacy"]["metadata_only"])

    def test_outcome_event_uses_metadata_only_aggregate_fields(self):
        feedback = experiments.routing_experiment_feedback_features(
            experiment_id="local-exp-secret",
            experiment_meta={
                "sampled": True,
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
                "mode": "shadow_candidate_pass_through",
                "counterfactual": True,
                "shadow_only": True,
            },
            routing_meta={
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "reason": "tool-result processing turn routed to Haiku",
            },
            comparison={
                "primary_output_chars": 17,
                "shadow_output_chars": 19,
                "primary_output_sha256": "sha256:primary",
                "shadow_output_sha256": "sha256:shadow",
                "output_similarity": 0.91,
                "passed_threshold": True,
            },
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )

        event = experiments.routing_experiment_outcome_event(feedback)

        assert_managed_egress_safe(event)
        self.assertEqual(event["schema"], "tokenclaw.routing_experiment_outcome_event.v1")
        self.assertEqual(event["candidate"]["mode"], "shadow_candidate_pass_through")
        self.assertTrue(event["candidate"]["counterfactual"])
        self.assertTrue(event["candidate"]["shadow_only"])
        self.assertEqual(event["outcome"]["mode"], "shadow_candidate_pass_through")
        self.assertEqual(event["source_surface"], "anthropic_messages")
        self.assertEqual(event["app_family"], "claude_code")
        self.assertEqual(event["workflow_phase"], "tool-execution")
        self.assertEqual(event["candidate"]["candidate_bucket"], "tool-result:sonnet->haiku")
        self.assertEqual(event["candidate"]["requested_model_family"], "sonnet")
        self.assertEqual(event["candidate"]["routed_model_family"], "haiku")
        self.assertEqual(event["candidate"]["shadow_model_family"], "sonnet")
        self.assertEqual(event["outcome"]["primary_status_class"], "2xx")
        self.assertEqual(event["outcome"]["shadow_status_class"], "2xx")
        self.assertEqual(event["outcome"]["output_similarity"], 0.91)
        self.assertEqual(event["outcome"]["primary_output_sha256"], "sha256:primary")
        self.assertTrue(event["privacy"]["metadata_only"])
        rendered = json.dumps(event, sort_keys=True)
        self.assertNotIn("claude-sonnet-4-6", rendered)
        self.assertNotIn("claude-haiku-4-5-20251001", rendered)
        self.assertNotIn("local-exp-secret", rendered)
        for forbidden_key in (
            '"messages"',
            '"provider_body"',
            '"raw_request"',
            '"raw_response"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
            '"tenant_id"',
        ):
            self.assertNotIn(forbidden_key, rendered)

    def test_disabled_managed_mode_queues_routing_experiment_policy_event(self):
        self._disable_managed_backing()
        feedback = experiments.routing_experiment_feedback_features(
            experiment_id="exp-1",
            experiment_meta={
                "sampled": True,
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
            },
            routing_meta={"category": "tool-result", "workflow_phase": "tool-execution"},
            comparison={
                "primary_output_chars": 17,
                "shadow_output_chars": 19,
                "primary_output_sha256": "sha256:primary",
                "shadow_output_sha256": "sha256:shadow",
                "output_similarity": 0.91,
                "passed_threshold": True,
            },
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )
        event = experiments.routing_experiment_outcome_event(feedback)

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                meta = asyncio.run(
                    queue_policy_event_feedback(
                        store,
                        event,
                        source_surface=experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
                        queue_when_disabled=True,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self.assertEqual(row["source_surface"], experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["schema"], "tokenclaw.routing_experiment_outcome_event.v1")
        assert_managed_egress_safe(payload)
        self.assertNotIn("claude-sonnet-4-6", row["payload_json"])
        self.assertNotIn("claude-haiku-4-5-20251001", row["payload_json"])

    def test_enabled_managed_mode_can_queue_routing_experiment_for_explicit_flush(self):
        event = experiments.routing_experiment_outcome_event({
            "schema": "tokenclaw.routing_experiment_feedback.v1",
            "experiment_id": "exp-1",
            "sampled": True,
            "status": "compared",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-haiku-4-5-20251001",
            "shadow_model": "claude-sonnet-4-6",
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "text_chars_bucket": "2k-8k",
            "primary_status_code": 200,
            "shadow_status_code": 200,
            "output_similarity": 0.91,
            "similarity_threshold": 0.86,
            "passed_threshold": True,
            "reason_codes": ["passed"],
        })

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
                os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
                meta = asyncio.run(
                    queue_policy_event_feedback(
                        store,
                        event,
                        source_surface=experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
                        queue_when_disabled=True,
                        flush_immediately=False,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued")
        self.assertEqual(row["source_surface"], experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)

    def _reload_with_promotion_fixture_policy(self, tmp_path: Path, *, daily_budget_usd: float = 10.0):
        config = tmp_path / "config"
        config.mkdir(exist_ok=True)
        (config / "routing_experiments.yaml").write_text(
            f"""
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: {daily_budget_usd}
providers:
  - openai
source_surfaces:
  - codex_turn
model_pairs:
  - requested_model: gpt-5-codex
    routed_model: gpt-5-mini
categories:
  - codex-turn
min_samples_for_confidence: 3
similarity_threshold: 0.86
""",
            encoding="utf-8",
        )
        os.chdir(tmp_path)
        return importlib.reload(experiments)

    def _log_shadow_promotion_sample(
        self,
        store: Store,
        *,
        idx: int,
        created_at: str | None = None,
        mode: str = "shadow_candidate_pass_through",
        passed: bool = True,
        shadow_status_code: int | None = 200,
        error: str | None = None,
        workflow_phase: str = "summary",
        fallback: bool = False,
        stream: bool = False,
        primary_cost_est_usd: float = 0.003,
        shadow_cost_est_usd: float = 0.001,
        shadow_routed_cost_est_usd: float | None = None,
    ) -> None:
        created_at = created_at or utc_now()
        similarity = 0.94 if passed else 0.7
        experiment_json = {
            "mode": mode,
            "counterfactual": mode == "shadow_candidate_pass_through",
            "shadow_only": mode == "shadow_candidate_pass_through",
            "workflow_phase": workflow_phase,
            "raw_prompt": "raw promotion secret",
            "request_id": "req-secret",
            "session_id": "session-secret",
            "file_path": "/tmp/secret.py",
            "cache_key": "cache-secret",
        }
        routing_json = {"workflow_phase": workflow_phase}
        if fallback:
            routing_json["fallback_reason"] = "rate_limited"
        store.log_routing_experiment(
            id=f"promotion-sample-{mode}-{idx}",
            call_id=f"promotion-call-{idx}",
            created_at=created_at,
            provider="openai",
            source_surface="codex_turn",
            stream=1 if stream else 0,
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            primary_model="gpt-5-codex",
            shadow_model="gpt-5-mini",
            category="codex-turn",
            routing_reason="sampled-shadow-candidate-pass-through",
            input_tokens_est=100,
            primary_status_code=200,
            shadow_status_code=shadow_status_code,
            primary_latency_ms=120,
            shadow_latency_ms=80,
            primary_output_chars=20,
            shadow_output_chars=18,
            primary_output_sha256=f"primary-{idx}",
            shadow_output_sha256=f"shadow-{idx}",
            output_similarity=similarity if shadow_status_code and shadow_status_code < 400 else None,
            passed_threshold=1 if passed else 0,
            primary_cost_est_usd=primary_cost_est_usd,
            shadow_cost_est_usd=shadow_cost_est_usd,
            shadow_routed_cost_est_usd=shadow_routed_cost_est_usd,
            routing_json=stable_json(routing_json),
            experiment_json=stable_json(experiment_json),
            error=error,
        )

    def _promotion_report_for_samples(self, *, sample_count: int = 3, **sample_kwargs):
        tmp = TemporaryDirectory()
        tmp_path = Path(tmp.name)
        manual = self._reload_with_promotion_fixture_policy(
            tmp_path,
            daily_budget_usd=sample_kwargs.pop("daily_budget_usd", 10.0),
        )
        store = Store(str(tmp_path / "tokenclaw.sqlite3"))
        try:
            for idx in range(sample_count):
                self._log_shadow_promotion_sample(store, idx=idx, **sample_kwargs)
            return manual.build_routing_experiment_report(store, limit=10)
        finally:
            store.conn.close()
            tmp.cleanup()

    def test_shadow_promotion_report_marks_healthy_candidate_promote_without_raw_fields(self):
        report = self._promotion_report_for_samples()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "promote")
        self.assertEqual(candidate["promotion"]["evidence_kind"], "shadow_pass_through")
        self.assertEqual(candidate["promotion"]["promotion_scope"], "stage_local_canary_from_shadow")
        self.assertFalse(candidate["promotion"]["canary_evidence"])
        self.assertIn("promotion-thresholds-met", candidate["promotion_reason_codes"])
        self.assertEqual(report["summary"]["promotion_ready_candidates"], 1)
        self.assertEqual(report["summary"]["promotion_verdict_counts"]["promote"], 1)
        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw promotion secret",
            "req-secret",
            "session-secret",
            "/tmp/secret.py",
            "cache-secret",
            '"raw_prompt"',
            '"request_id"',
            '"session_id"',
            '"file_path"',
            '"cache_key"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_promotion_uses_counterfactual_routed_cost_not_uncached_probe(self):
        # The shadow probe is a fresh, uncached call, so on cached traffic it costs
        # more than the cached primary (e.g. $0.96 vs $0.25) — which used to reject
        # the candidate as "shadow-more-expensive" even though live routing to the
        # cheaper model gets the same caching and is actually cheaper. The promotion
        # cost delta must use the counterfactual routed cost.
        report = self._promotion_report_for_samples(
            primary_cost_est_usd=0.25,
            shadow_cost_est_usd=0.96,  # uncached probe — more expensive
            shadow_routed_cost_est_usd=0.15,  # what the route would actually cost
        )

        candidate = report["candidates"][0]
        self.assertGreater(candidate["cost_delta_usd"], 0)
        self.assertNotIn("shadow-more-expensive", candidate["promotion_reason_codes"])
        self.assertEqual(candidate["promotion_verdict"], "promote")

    def test_promotion_rejects_when_counterfactual_route_is_more_expensive(self):
        report = self._promotion_report_for_samples(
            primary_cost_est_usd=0.25,
            shadow_cost_est_usd=0.10,
            shadow_routed_cost_est_usd=0.40,  # route genuinely costs more
        )

        candidate = report["candidates"][0]
        self.assertLess(candidate["cost_delta_usd"], 0)
        self.assertEqual(candidate["promotion_verdict"], "reject")
        self.assertIn("shadow-more-expensive", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_needs_more_samples_for_low_count(self):
        report = self._promotion_report_for_samples(sample_count=2)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "needs_more_samples")
        self.assertIn("insufficient-samples", candidate["promotion_reason_codes"])
        self.assertIn("insufficient-compared-samples", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_holds_stale_evidence(self):
        stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        report = self._promotion_report_for_samples(created_at=stale)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "hold")
        self.assertIn("stale-evidence", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_rejects_low_similarity_pass_rate(self):
        report = self._promotion_report_for_samples(passed=False)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "reject")
        self.assertIn("below-similarity-pass-rate", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_rejects_high_shadow_error_rate(self):
        report = self._promotion_report_for_samples(shadow_status_code=500, error="shadow failed")

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "reject")
        self.assertIn("shadow-error-rate-high", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_holds_budget_exhaustion(self):
        report = self._promotion_report_for_samples(daily_budget_usd=0.0)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "hold")
        self.assertIn("daily-budget-exhausted", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_separates_shadow_and_applied_modes(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual = self._reload_with_promotion_fixture_policy(tmp_path)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx, mode="shadow_candidate_pass_through")
                    self._log_shadow_promotion_sample(store, idx=idx + 10, mode="applied_routed_down")
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        modes = {candidate["mode"]: candidate for candidate in report["candidates"]}
        self.assertEqual(set(modes), {"shadow_candidate_pass_through", "applied_routed_down"})
        self.assertFalse(modes["shadow_candidate_pass_through"]["promotion"]["canary_evidence"])
        self.assertTrue(modes["applied_routed_down"]["promotion"]["canary_evidence"])

    def test_shadow_promotion_report_separates_streaming_scope(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual = self._reload_with_promotion_fixture_policy(tmp_path)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx, stream=True)
                    self._log_shadow_promotion_sample(store, idx=idx + 10, stream=False)
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        streams = {candidate["stream"] for candidate in report["candidates"]}
        self.assertEqual(streams, {True, False})
        for candidate in report["candidates"]:
            self.assertEqual(candidate["promotion_verdict"], "promote")

    def test_routing_experiment_report_cli_and_stats_full_expose_promotion_verdicts(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._reload_with_promotion_fixture_policy(tmp_path)
            db_path = str(tmp_path / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx)
                out = StringIO()
                code = cli.routing_experiment_report_cli(["--db", db_path], stdout=out)
                payload = json.loads(out.getvalue())
                full = asyncio.run(stats.stats_full(store))
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(payload["candidates"][0]["promotion_verdict"], "promote")
        self.assertEqual(full["routing_experiment_report"]["candidates"][0]["promotion_verdict"], "promote")
        self.assertEqual(full["summary"]["routing_experiment_promotion_ready_candidates"], 1)

    def test_dashboard_added_candidate_lifecycle_outcomes_are_metadata_only(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            candidate_id = "dashboard-anthropic-claude-opus-5-0-to-claude-sonnet-4-6-chat-fixture"
            (config / "routing_experiments.yaml").write_text(
                f"""
enabled: true
mode: applied_routed_down
sample_rate: 1.0
daily_budget_usd: 10.0
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
categories:
  - chat
routing_candidates:
  - candidate_id: {candidate_id}
    candidate_source: dashboard-recent-call
    requested_model: claude-opus-5-0
    routed_model: claude-sonnet-4-6
    provider: anthropic
    source_surface: anthropic_messages
    app_family: claude_code
    category: chat
    stream: false
    max_text_chars: 32000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                base_experiment = {
                    "schema": "tokenclaw.routing_experiment_decision.v1",
                    "candidate_id": candidate_id,
                    "candidate_source": "dashboard-recent-call",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "requested_model": "claude-opus-5-0",
                    "routed_model": "claude-sonnet-4-6",
                    "category": "chat",
                    "status": "selected",
                    "sampled": True,
                    "mode": "applied_routed_down",
                    "privacy": {
                        "metadata_only": True,
                        "raw_prompts_included": False,
                        "provider_bodies_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                    },
                }
                applied_routing = {
                    "routing_experiment": {
                        **base_experiment,
                        "canary_cohort": "canary_applied",
                        "reason": "selected-canary",
                    }
                }
                holdout_routing = {
                    "routing_experiment": {
                        **base_experiment,
                        "canary_cohort": "canary_holdout",
                        "reason": "selected-holdout",
                    }
                }
                store.log_call(
                    id="local-applied-call-id-not-in-output",
                    created_at="2026-06-22T08:00:00+00:00",
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-opus-5-0",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=900,
                    input_tokens_est=2400,
                    actual_input_tokens=2400,
                    output_tokens_est=200,
                    actual_output_tokens=200,
                    cost_baseline_usd=0.09,
                    cost_est_usd=0.03,
                    category="chat",
                    retry_count=0,
                    routing_json=stable_json(applied_routing),
                    error=None,
                )
                store.log_call(
                    id="local-holdout-call-id-not-in-output",
                    created_at="2026-06-22T08:05:00+00:00",
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-opus-5-0",
                    routed_model="claude-opus-5-0",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1100,
                    input_tokens_est=2500,
                    actual_input_tokens=2500,
                    output_tokens_est=210,
                    actual_output_tokens=210,
                    cost_baseline_usd=0.095,
                    cost_est_usd=0.095,
                    category="chat",
                    retry_count=0,
                    routing_json=stable_json(holdout_routing),
                    error=None,
                )
                store.log_routing_experiment(
                    id="shadow-row-not-in-output",
                    call_id="shadow-call-not-in-output",
                    created_at="2026-06-22T08:10:00+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    stream=0,
                    requested_model="claude-opus-5-0",
                    routed_model="claude-sonnet-4-6",
                    primary_model="claude-opus-5-0",
                    shadow_model="claude-sonnet-4-6",
                    category="chat",
                    routing_reason="sampled-routed-down-call",
                    input_tokens_est=2400,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=1100,
                    shadow_latency_ms=900,
                    primary_output_chars=10,
                    shadow_output_chars=10,
                    primary_output_sha256="primary",
                    shadow_output_sha256="shadow",
                    output_similarity=0.99,
                    passed_threshold=1,
                    primary_cost_est_usd=0.09,
                    shadow_cost_est_usd=0.03,
                    routing_json=stable_json(applied_routing),
                    experiment_json=stable_json(base_experiment),
                )
                lifecycle = manual.build_routing_experiment_lifecycle_outcomes(store, limit=10, window_hours=0)
                report = manual.build_routing_experiment_report(store, limit=10, window_hours=0)
            finally:
                store.conn.close()

        self.assertEqual(lifecycle["schema"], "tokenclaw.routing_experiment_lifecycle_outcomes.v1")
        self.assertEqual(lifecycle["summary"]["applied_count"], 1)
        self.assertEqual(lifecycle["summary"]["holdout_count"], 1)
        self.assertEqual(lifecycle["summary"]["matched_count"], 2)
        self.assertEqual(lifecycle["summary"]["observed_saved_usd"], 0.06)
        outcome = lifecycle["outcomes"][0]
        self.assertEqual(outcome["status"], "coverage-ready")
        self.assertTrue(outcome["fingerprint"].startswith("routing-candidate:"))
        self.assertEqual(outcome["candidate_source"], "dashboard-recent-call")
        self.assertEqual(outcome["applied_count"], 1)
        self.assertEqual(outcome["holdout_count"], 1)
        self.assertEqual(outcome["safety_stop_count"], 0)
        self.assertEqual(outcome["rollback_count"], 0)
        self.assertEqual(outcome["observed_savings_per_1000_calls_usd"], 60.0)
        self.assertEqual(outcome["projected_savings_per_1000_calls_usd"], 30.0)
        self.assertEqual(outcome["regression_deltas"]["error_rate_delta"], 0.0)
        self.assertEqual(outcome["regression_deltas"]["retry_rate_delta"], 0.0)
        self.assertTrue(outcome["coverage"]["has_applied_coverage"])
        self.assertTrue(outcome["coverage"]["has_holdout_coverage"])
        self.assertTrue(outcome["privacy"]["metadata_only"])
        self.assertFalse(outcome["privacy"]["individual_candidate_ids_included"])
        self.assertFalse(outcome["privacy"]["request_ids_included"])
        rendered = stable_json(lifecycle)
        self.assertNotIn(candidate_id, rendered)
        self.assertNotIn("local-applied-call-id-not-in-output", rendered)
        self.assertIn("lifecycle_outcomes", report)
        self.assertEqual(report["summary"]["routing_lifecycle_applied_count"], 1)
        self.assertEqual(report["summary"]["routing_lifecycle_holdout_count"], 1)


if __name__ == "__main__":
    unittest.main()
