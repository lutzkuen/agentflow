import unittest
import base64
import importlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

import tokenclaw.cache as cache_module
from tokenclaw import cli
from tokenclaw.managed_egress import ManagedEgressBlocked, RAW_FEATURE_KEYS, assert_managed_egress_safe
from tokenclaw.policy_bundle import apply_policy_bundle, validate_policy_bundle
from tokenclaw.policy_events import recent_policy_events
from tokenclaw.store import Store, stable_json


class CacheDecisionMetaTest(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_CACHE",
        "TOKENCLAW_CACHE_TOOL_CALLS",
        "TOKENCLAW_SEMANTIC_CACHE",
        "TOKENCLAW_SEMANTIC_THRESHOLD",
        "TOKENCLAW_CACHE_RULES",
        "TOKENCLAW_PROVIDER",
        "TOKENCLAW_ANTHROPIC_UPSTREAM",
        "TOKENCLAW_OPENAI_UPSTREAM",
        "TOKENCLAW_CACHE_NAMESPACE",
        "TOKENCLAW_CACHE_FILE_WATCH",
        "TOKENCLAW_CACHE_WATCH_ROOT",
        "TOKENCLAW_CACHE_WATCH_MAX_PATHS",
        "TOKENCLAW_CACHE_CAPTURE_CANDIDATES",
        "TOKENCLAW_CACHE_TTL_SECONDS",
        "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP",
        "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP_WINDOW",
        "TOKENCLAW_POLICY_EVENTS",
        "TOKENCLAW_POLICY_EVENTS_LOG",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name
        importlib.reload(cache_module)

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(cache_module)

    def test_cache_hit_metadata_has_explicit_status_and_source(self):
        meta = cache_module.cache_decision_meta(
            "hit",
            "exact-match",
            hit_type="exact",
            exact_enabled=True,
            semantic_enabled=False,
        )

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "hit")
        self.assertEqual(meta["reason"], "exact-match")
        self.assertEqual(meta["hit_type"], "exact")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_packaged_haiku_short_completion_streaming_rule_is_narrow_and_warm(self):
        rules = list(cache_module.CACHE_PATTERN_RULES)
        [rule] = [
            item for item in rules
            if item.get("id") == "local-haiku-short-completion-streaming-cache"
        ]

        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["conditions"]["source_surface"], "anthropic_messages")
        self.assertEqual(rule["conditions"]["app_family"], "claude_code")
        self.assertEqual(rule["conditions"]["model_pattern"], "haiku")
        self.assertEqual(rule["conditions"]["category"], "short-completion")
        self.assertTrue(rule["conditions"]["stream"])
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertEqual(rule["conditions"]["cacheability_bucket"], "high")
        self.assertTrue(rule["conditions"]["static_information_hint"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertTrue(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["min_call_count"], 5)

    def test_packaged_openai_cache_replay_rule_rolls_back_stale_49_row_cohort(self):
        rules = list(cache_module.CACHE_PATTERN_RULES)
        [rule] = [
            item for item in rules
            if item.get("id") == "local-openai-cache-replay-canary-ae8404ee817f89f4"
        ]

        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["candidate_id"], "request-shape-cache-replay:responses:chat:8e210a2f5680d16d")
        raw_policy = yaml.safe_load((Path(cache_module.__file__).with_name("cache_rules.yaml")).read_text())
        [raw_rule] = [
            item for item in raw_policy["pattern_rules"]
            if item.get("id") == "local-openai-cache-replay-canary-ae8404ee817f89f4"
        ]
        self.assertEqual(raw_rule["disabled_reason"], "stale-no-canary-traffic")
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertEqual(rule["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(rule["conditions"]["token_bucket"], "500_2k_tokens")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["canary_unit"], "request_fingerprint")
        self.assertEqual(rule["graduation"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(rule["graduation"]["sample_count"], 49)
        self.assertEqual(rule["graduation"]["projected_hits"], 48)
        self.assertEqual(rule["graduation"]["projected_savings_usd"], 0.102518)

        features = {
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "chat",
            "workflow_phase": "chat",
            "text_bucket": "2k_8k_chars",
            "token_bucket": "500_2k_tokens",
            "has_tools": False,
            "stream": False,
            "replayability_level": "features_only",
            "request_fingerprint": "sha256:" + "0" * 64,
            "raw_pattern_strings_included": False,
        }
        _, _, meta = cache_module.cache_lookup_meta(has_tool_blocks=False, pattern_features=features)
        skip_reasons = (meta.get("pattern_rules") or {}).get("skip_reasons") or []
        self.assertIn(
            {
                "rule_id": "local-openai-cache-replay-canary-ae8404ee817f89f4",
                "reason": "disabled",
            },
            skip_reasons,
        )

    def test_openai_cache_replay_hit_recovery_smoke_demonstrates_second_hit(self):
        from tokenclaw.cache_smoke import build_cache_replay_hit_recovery_smoke

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                result = build_cache_replay_hit_recovery_smoke(store)
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "tokenclaw.cache_replay_hit_recovery_smoke.v1")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "no-synthetic-canary-applied-fingerprint")
        self.assertEqual(result["blocker_codes"], ["canary-applied-cohort-unavailable"])
        self.assertEqual(result["target_rule_id"], "local-openai-cache-replay-canary-ae8404ee817f89f4")
        self.assertEqual(result["target_shape"]["provider_family"], "openai")
        self.assertEqual(result["target_shape"]["source_surface"], "openai_responses")
        self.assertEqual(result["target_shape"]["endpoint"], "responses")
        self.assertEqual(result["target_shape"]["category"], "chat")
        self.assertEqual(result["target_shape"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(result["target_shape"]["token_bucket"], "500_2k_tokens")
        self.assertFalse(result["target_shape"]["has_tools"])
        self.assertFalse(result["target_shape"]["stream"])
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertTrue(result["privacy"]["synthetic_only"])
        self.assertFalse(result["privacy"]["provider_calls_made"])
        self.assertFalse(result["privacy"]["managed_server_calls_made"])
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["raw_responses_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["request_fingerprints_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("AgentFlow deterministic OpenAI cache replay", rendered)
        self.assertNotIn("tokenclaw-cache-replay-hit-recovery-smoke-", rendered)
        self.assertNotIn("tokenclaw-cache-replay-hit-recovery-smoke-session", rendered)

    def test_cache_smoke_diagnostic_includes_isolated_hit_recovery_without_mutating_store(self):
        from tokenclaw.cache_smoke import build_cache_smoke_diagnostic

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                before = store.conn.execute("select count(*) as c from cache").fetchone()["c"]
                result = build_cache_smoke_diagnostic(store, limit=1, scan_limit=1)
                after = store.conn.execute("select count(*) as c from cache").fetchone()["c"]
            finally:
                store.conn.close()

        self.assertEqual(before, 0)
        self.assertEqual(after, 0)
        smoke = result["cache_replay_hit_recovery_smoke"]
        self.assertEqual(smoke["schema"], "tokenclaw.cache_replay_hit_recovery_smoke.v1")
        self.assertEqual(smoke["status"], "blocked")
        self.assertEqual(smoke["reason"], "no-synthetic-canary-applied-fingerprint")
        self.assertEqual(smoke["blocker_codes"], ["canary-applied-cohort-unavailable"])
        self.assertTrue(result["privacy"]["synthetic_hit_recovery_included"])

    def test_tool_requests_are_skipped_when_tool_cache_disabled(self):
        can_exact, can_semantic, meta = cache_module.cache_lookup_meta(has_tool_blocks=True)

        self.assertFalse(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "tools-disabled")
        self.assertTrue(meta["enabled"])
        self.assertFalse(meta["tool_cache_enabled"])

    def test_non_tool_requests_report_exact_miss_by_default(self):
        can_exact, can_semantic, meta = cache_module.cache_lookup_meta(has_tool_blocks=False)

        self.assertTrue(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "miss")
        self.assertEqual(meta["reason"], "exact-miss")

    def test_cache_key_is_namespaced_by_provider_upstream_and_namespace(self):
        body = {"model": "same-model", "messages": [{"role": "user", "content": "same"}]}

        anthropic = cache_module.cache_key_for(
            body,
            "/v1/messages",
            provider="anthropic",
            upstream="https://api.anthropic.com",
            namespace="project-a",
        )
        openai = cache_module.cache_key_for(
            body,
            "/v1/messages",
            provider="openai",
            upstream="https://api.openai.com",
            namespace="project-a",
        )
        other_namespace = cache_module.cache_key_for(
            body,
            "/v1/messages",
            provider="anthropic",
            upstream="https://api.anthropic.com",
            namespace="project-b",
        )
        other_upstream = cache_module.cache_key_for(
            body,
            "/v1/messages",
            provider="anthropic",
            upstream="https://staging.anthropic.example",
            namespace="project-a",
        )

        self.assertNotEqual(anthropic, openai)
        self.assertNotEqual(anthropic, other_namespace)
        self.assertNotEqual(anthropic, other_upstream)
        self.assertEqual(
            anthropic,
            cache_module.cache_key_for(
                body,
                "/v1/messages",
                provider="anthropic",
                upstream="https://api.anthropic.com/",
                namespace="project-a",
            ),
        )

    def test_cache_key_uses_environment_namespace_by_default(self):
        body = {"model": "same-model", "messages": [{"role": "user", "content": "same"}]}
        os.environ["TOKENCLAW_CACHE_NAMESPACE"] = "env-project"
        os.environ["TOKENCLAW_PROVIDER"] = "openai"
        os.environ["TOKENCLAW_OPENAI_UPSTREAM"] = "https://openai.example"

        key_from_env = cache_module.cache_key_for(body, "/v1/responses")
        explicit_key = cache_module.cache_key_for(
            body,
            "/v1/responses",
            provider="openai",
            upstream="https://openai.example",
            namespace="env-project",
        )

        self.assertEqual(key_from_env, explicit_key)

    def test_config_cache_rules_can_change_cache_behavior_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                """
exact_cache:
  enabled: false
  cache_tool_calls: false
semantic_cache:
  enabled: true
  threshold: 0.82
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)

            can_exact, can_semantic, meta = manual.cache_lookup_meta(has_tool_blocks=False)

            self.assertFalse(can_exact)
            self.assertTrue(can_semantic)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "semantic-miss")
            self.assertEqual(meta["policy_source"], "local-manual")
            # rule_path is built from Path.cwd(), which getcwd() canonicalizes;
            # resolve the expected path too so macOS /var -> /private/var matches.
            self.assertEqual(meta["rule_path"], str((config / "cache_rules.yaml").resolve()))
            self.assertEqual(meta["semantic_threshold"], 0.82)

    def test_config_cache_rules_can_enable_exact_tool_cache_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                """
exact_cache:
  enabled: true
  cache_tool_calls: true
semantic_cache:
  enabled: false
  threshold: 0.95
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)

            can_exact, can_semantic, meta = manual.cache_lookup_meta(has_tool_blocks=True)

            self.assertTrue(can_exact)
            self.assertFalse(can_semantic)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "exact-miss")
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertTrue(meta["tool_cache_enabled"])
            self.assertTrue(meta["file_watch_enabled"])

    def test_reviewed_bundle_applies_safe_managed_cache_pattern_rule(self):
        pattern_hash = "sha256:" + "a" * 64
        unsafe_hash = "sha256:" + "b" * 64
        with TemporaryDirectory() as tmp:
            exported = io.StringIO()
            self.assertEqual(cli.policy_export_cli([], stdout=exported), 0)
            bundle = json.loads(exported.getvalue())
            cache_policy = bundle["policies"]["cache"]
            cache_policy["policy_source"] = "managed-recommended"
            cache_policy["pattern_rules"] = [{
                "id": "reviewed-cache-pattern",
                "enabled": True,
                "policy_source": "managed-recommended",
                "candidate_id": "cache-candidate-safe",
                "conditions": {
                    "pattern_hashes": [pattern_hash],
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "has_tools": True,
                    "stream": False,
                    "replayability_level": "local-exact-response",
                },
                "action": {
                    "type": "exact_cache_pattern",
                    "allow_tool_calls": True,
                    "safe_invalidation": True,
                    "estimated_saved_cost_usd": 0.012,
                },
            }]
            cache_policy["recommendation"] = {
                "policy_source": "managed-recommended",
                "omitted_candidates": [{
                    "candidate_id": "cache-candidate-unsafe",
                    "policy_source": "managed-recommended",
                    "pattern_hash": unsafe_hash,
                    "omission_reasons": ["replay-safety-gate-failed"],
                    "local_action_requirements": {"expected_policy_section": "cache"},
                }],
            }

            result = apply_policy_bundle(bundle, config_dir=tmp)

            self.assertTrue(result["ok"], result)
            cache_rules_path = Path(tmp) / "cache_rules.yaml"
            written = yaml.safe_load(cache_rules_path.read_text(encoding="utf-8"))
            self.assertEqual(written["pattern_rules"][0]["candidate_id"], "cache-candidate-safe")
            self.assertNotIn("cache-candidate-unsafe", cache_rules_path.read_text(encoding="utf-8"))

            os.environ["TOKENCLAW_CACHE_RULES"] = str(cache_rules_path)
            manual = importlib.reload(cache_module)
            can_exact, can_semantic, meta = manual.cache_lookup_meta(
                has_tool_blocks=True,
                pattern_features={
                    "pattern_hashes": [pattern_hash],
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "workflow_phase": "tool-result",
                    "text_bucket": "2k_8k_chars",
                    "token_bucket": "1k_4k_tokens",
                    "has_tools": True,
                    "stream": False,
                    "raw_pattern_strings_included": False,
                },
            )

            self.assertTrue(can_exact)
            self.assertFalse(can_semantic)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "exact-pattern-miss")
            self.assertEqual(meta["pattern_rule"]["rule_id"], "reviewed-cache-pattern")
            self.assertEqual(meta["pattern_rule"]["candidate_id"], "cache-candidate-safe")
            self.assertEqual(meta["pattern_rule"]["policy_source"], "managed-recommended")
            self.assertEqual(meta["pattern_rule"]["matched_hashes"], [pattern_hash])
            serialized = json.dumps(meta, sort_keys=True)
            for forbidden in ("raw_prompt", "provider_body", "transcript", "tool_payload", "cache-key"):
                self.assertNotIn(forbidden, serialized)

    def test_cache_pattern_canary_rollout_is_deterministic(self):
        pattern_hash = "sha256:" + "e" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "cache_rules.yaml"
            rules_path.write_text(
                f"""
exact_cache:
  enabled: true
  cache_tool_calls: false
pattern_rules:
  - id: reviewed-cache-canary
    enabled: true
    policy_source: managed-recommended
    candidate_id: cache-candidate-canary
    conditions:
      pattern_hashes:
        - {pattern_hash}
      source_surface: anthropic_messages
      app_family: claude_code
      category: tool-result
      workflow_phase: tool-result
      has_tools: true
      stream: false
    rollout:
      schema: tokenclaw.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: 0.10
      canary_salt: sha256:0000000000000000000000000000000000000000000000000000000000000004
      canary_unit: request_fingerprint
    action:
      type: exact_cache_pattern
      allow_tool_calls: true
      safe_invalidation: true
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)
            features = {
                "pattern_hashes": [pattern_hash],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "tool-result",
                "workflow_phase": "tool-result",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "1k_4k_tokens",
                "has_tools": True,
                "stream": False,
                "raw_pattern_strings_included": False,
            }

            can_exact_1, _, meta_1 = manual.cache_lookup_meta(has_tool_blocks=True, pattern_features=features)
            can_exact_2, _, meta_2 = manual.cache_lookup_meta(has_tool_blocks=True, pattern_features=features)

            self.assertTrue(can_exact_1)
            self.assertTrue(can_exact_2)
            self.assertEqual(meta_1["reason"], "exact-pattern-miss")
            self.assertEqual(meta_1["pattern_rule"]["candidate_id"], "cache-candidate-canary")
            self.assertEqual(meta_1["pattern_rule"]["canary"]["status"], "applied")
            self.assertEqual(meta_1["pattern_rule"]["canary"], meta_2["pattern_rule"]["canary"])

            rules_path.write_text(
                rules_path.read_text(encoding="utf-8").replace(
                    "sha256:0000000000000000000000000000000000000000000000000000000000000004",
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                ),
                encoding="utf-8",
            )
            holdout = importlib.reload(cache_module)
            can_exact_holdout, _, holdout_meta = holdout.cache_lookup_meta(
                has_tool_blocks=True,
                pattern_features=features,
            )

            self.assertFalse(can_exact_holdout)
            self.assertEqual(holdout_meta["status"], "skipped")
            self.assertEqual(holdout_meta["reason"], "tools-disabled")
            holdout_reason = holdout_meta["pattern_rules"]["skip_reasons"][-1]
            self.assertEqual(holdout_reason["reason"], "canary_holdout")
            self.assertEqual(holdout_reason["candidate_id"], "cache-candidate-canary")
            self.assertEqual(holdout_reason["canary"]["status"], "holdout")

            rules_path.write_text("exact_cache:\n  enabled: true\n  cache_tool_calls: false\n", encoding="utf-8")
            default = importlib.reload(cache_module)
            can_exact_default, _, default_meta = default.cache_lookup_meta(has_tool_blocks=True, pattern_features=features)
            self.assertFalse(can_exact_default)
            self.assertEqual(default_meta["pattern_rules"]["configured_count"], 0)

    def test_cache_pattern_safety_stop_bypasses_failed_canary(self):
        pattern_hash = "sha256:" + "9" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = str(tmp_path / "policy_events.jsonl")
            (config / "cache_rules.yaml").write_text(
                f"""
exact_cache:
  enabled: true
  cache_tool_calls: false
pattern_rules:
  - id: reviewed-cache-safety-stop
    enabled: true
    policy_source: managed-recommended
    candidate_id: cache-candidate-safety-stop
    conditions:
      pattern_hashes:
        - {pattern_hash}
      source_surface: anthropic_messages
      app_family: claude_code
      category: tool-result
      workflow_phase: tool-result
      has_tools: true
      stream: false
    rollout:
      schema: tokenclaw.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: 1.0
      canary_salt: local-cache-safety-stop-test
      canary_unit: request_fingerprint
      min_outcome_samples: 2
      rollback_threshold: 0.5
    action:
      type: exact_cache_pattern
      allow_tool_calls: true
      safe_invalidation: true
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            features = {
                "pattern_hashes": [pattern_hash],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "tool-result",
                "workflow_phase": "tool-result",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "1k_4k_tokens",
                "has_tools": True,
                "stream": False,
                "raw_pattern_strings_included": False,
            }

            can_exact, _, healthy_meta = manual.cache_lookup_meta(
                has_tool_blocks=True,
                pattern_features=features,
                store_obj=store,
            )
            self.assertTrue(can_exact)
            self.assertEqual(healthy_meta["pattern_rule"]["rule_id"], "reviewed-cache-safety-stop")

            for index in range(2):
                store.log_call(
                    id=f"failed-cache-canary-{index}",
                    created_at=f"2026-06-09T00:1{index}:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=500,
                    latency_ms=100,
                    input_tokens_est=1000,
                    output_tokens_est=0,
                    actual_input_tokens=1000,
                    actual_output_tokens=0,
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.0,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"category": "tool-result"}),
                    cache_json=stable_json({
                        "status": "miss",
                        "reason": "exact-pattern-miss",
                        "policy_source": "local-default",
                        "pattern_rule": {
                            "rule_id": "reviewed-cache-safety-stop",
                            "candidate_id": "cache-candidate-safety-stop",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [pattern_hash],
                            "canary": {
                                "schema": "tokenclaw.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": True,
                                "status": "applied",
                                "cohort": "canary_applied",
                            },
                        },
                    }),
                    error="upstream failed",
                    request_json=None,
                    response_json=None,
                    session_id="cache-safety-stop",
                    category="tool-result",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    provider="anthropic",
                )

            stopped_exact, _, stopped_meta = manual.cache_lookup_meta(
                has_tool_blocks=True,
                pattern_features=features,
                store_obj=store,
            )
            self.assertFalse(stopped_exact)
            reasons = stopped_meta["pattern_rules"]["skip_reasons"]
            safety_reason = next(item for item in reasons if item["reason"] == "local-canary-safety-stop")
            self.assertEqual(safety_reason["candidate_id"], "cache-candidate-safety-stop")
            self.assertEqual(safety_reason["safety_stop"]["sample_count"], 2)
            self.assertEqual(safety_reason["safety_stop"]["error_count"], 2)
            self.assertEqual(safety_reason["canary"]["status"], "applied")
            events = recent_policy_events(limit=5)["events"]
            self.assertEqual(events[0]["action"], "pattern-canary-safety-stop")
            self.assertEqual(events[0]["details"]["policy_section"], "cache")

    def test_cache_pattern_validation_rejects_unsafe_tool_rule(self):
        exported = io.StringIO()
        self.assertEqual(cli.policy_export_cli([], stdout=exported), 0)
        bundle = json.loads(exported.getvalue())
        bundle["policies"]["cache"]["pattern_rules"] = [{
            "id": "unsafe-tool-cache-pattern",
            "policy_source": "managed-recommended",
            "candidate_id": "cache-candidate-unsafe",
            "conditions": {
                "pattern_hash": "sha256:" + "c" * 64,
                "has_tools": True,
                "replayability_level": "local-exact-response",
            },
            "action": {
                "type": "exact_cache_pattern",
                "allow_tool_calls": True,
            },
        }]

        validation = validate_policy_bundle(bundle)

        self.assertFalse(validation["ok"])
        self.assertIn(
            "$.policies.cache.pattern_rules[0].action.safe_invalidation",
            {error["path"] for error in validation["errors"]},
        )

    def test_file_dependency_snapshots_capture_paths_under_watch_root(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('hello')\n", encoding="utf-8")
            os.chdir(tmp_path)

            snapshots = cache_module.cache_file_dependency_snapshots({
                "messages": [
                    {
                        "role": "user",
                        "content": "Read src/example.py:12 and ./missing.txt. Ignore https://example.com/a.py",
                    }
                ]
            })

            by_name = {Path(item["path"]).name: item for item in snapshots}
            self.assertEqual(by_name["example.py"]["exists"], True)
            self.assertEqual(by_name["example.py"]["size"], watched.stat().st_size)
            self.assertEqual(by_name["missing.txt"]["exists"], False)
            self.assertNotIn("a.py", by_name)

    def test_file_dependency_snapshots_skip_unexpandable_home_paths(self):
        original_expanduser = cache_module.Path.expanduser

        def fail_home_expand(path):
            if str(path).startswith("~"):
                raise RuntimeError("Could not determine home directory.")
            return original_expanduser(path)

        with patch.object(cache_module.Path, "expanduser", fail_home_expand):
            snapshots = cache_module.cache_file_dependency_snapshots({
                "messages": [
                    {
                        "role": "user",
                        "content": "Read ~/private.txt and ./relative.txt",
                    }
                ]
            })

        self.assertEqual([Path(item["path"]).name for item in snapshots], ["relative.txt"])

    def test_file_dependency_snapshots_can_be_disabled_by_policy(self):
        os.environ["TOKENCLAW_CACHE_FILE_WATCH"] = "0"
        disabled = importlib.reload(cache_module)

        snapshots = disabled.cache_file_dependency_snapshots({
            "messages": [{"role": "user", "content": "Read tokenclaw/cache.py"}]
        })

        self.assertEqual(snapshots, [])

    def test_file_dependency_audit_reports_cap_exceeded_without_paths(self):
        os.environ["TOKENCLAW_CACHE_WATCH_MAX_PATHS"] = "1"
        capped = importlib.reload(cache_module)
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
            (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
            os.chdir(tmp_path)

            audit = capped.cache_file_dependency_audit({
                "messages": [{"role": "user", "content": "Read ./a.txt and ./b.txt"}],
            })
            snapshots = capped.cache_file_dependency_snapshots({
                "messages": [{"role": "user", "content": "Read ./a.txt and ./b.txt"}],
            })

        self.assertTrue(audit["cap_exceeded"])
        self.assertFalse(audit["cap_trimmed"])
        self.assertEqual(audit["invalidation_reason"], "dependency-cap-exceeded")
        self.assertEqual(audit["dependency_capture_reason"], "dependency-cap-exceeded")
        self.assertEqual(audit["snapshot_count"], 1)
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(audit["paths_included"])
        self.assertNotIn("a.txt", json.dumps(audit))

    def test_tool_result_dependency_capture_dedupes_repeated_mentions_before_cap(self):
        os.environ["TOKENCLAW_CACHE_WATCH_MAX_PATHS"] = "3"
        compact = importlib.reload(cache_module)
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (tmp_path / "src" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
            os.chdir(tmp_path)
            repeated = "\n".join(
                f"line {idx}: ./src/app.py:1 ./src/util.py:2 progress {idx}/100"
                for idx in range(80)
            )
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": repeated}],
                    }
                ],
            }

            snapshots = compact.cache_file_dependency_snapshots(body)
            audit = compact.cache_file_dependency_audit(body)

        self.assertEqual(audit["snapshot_count"], 2)
        self.assertEqual(len(snapshots), 2)
        self.assertFalse(audit["cap_exceeded"])
        self.assertTrue(audit["cap_trimmed"])
        self.assertIsNone(audit["invalidation_reason"])
        self.assertEqual(audit["dependency_capture_reason"], "dependency-cap-trimmed")
        self.assertEqual(audit["distinct_candidate_path_count_bucket"], "2_5")
        self.assertEqual(audit["raw_candidate_path_count_bucket"], "128_plus")
        self.assertTrue(audit["safe_invalidation_evidence"])
        self.assertTrue(audit["file_dependency_evidence_available"])
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn("src/app.py", rendered)
        self.assertNotIn("src/util.py", rendered)
        self.assertNotIn(str(tmp_path), rendered)

    def test_tool_result_dependency_capture_ignores_noisy_shell_fragments(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            os.chdir(tmp_path)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "pytest 42/100 [====] 3/7 ok https://example.test/a/b ./src/main.py:9",
                            }
                        ],
                    }
                ],
            }

            snapshots = cache_module.cache_file_dependency_snapshots(body)
            audit = cache_module.cache_file_dependency_audit(body)

        self.assertEqual(audit["snapshot_count"], 1)
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(audit["cap_exceeded"])
        self.assertFalse(audit["cap_trimmed"])
        self.assertEqual(audit["dependency_capture_reason"], "complete")
        self.assertTrue(audit["safe_invalidation_evidence"])

    def test_tool_result_dependency_capture_records_stable_bare_file_names_without_paths(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
            (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
            os.chdir(tmp_path)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Reviewed pyproject.toml and README.md; ignore example.com prose.",
                            }
                        ],
                    }
                ],
            }

            snapshots = cache_module.cache_file_dependency_snapshots(body)
            audit = cache_module.cache_file_dependency_audit(body)

        self.assertEqual(audit["snapshot_count"], 2)
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(audit["safe_invalidation_evidence"])
        self.assertTrue(audit["file_dependency_evidence_available"])
        self.assertEqual(audit["candidate_path_count_bucket"], "2_5")
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn("pyproject.toml", rendered)
        self.assertNotIn("README.md", rendered)
        self.assertNotIn(str(tmp_path), rendered)

    def test_file_dependency_capture_ignores_prose_slash_fragments_around_stable_files(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            os.chdir(tmp_path)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Read src/main.py. The answer should consider emotional/contextual "
                            "preferences/facts and reminding/supporting notes, but those are prose."
                        ),
                    }
                ],
            }

            snapshots = cache_module.cache_file_dependency_snapshots(body)
            audit = cache_module.cache_file_dependency_audit(body)

        self.assertEqual(audit["snapshot_count"], 1)
        self.assertEqual(len(snapshots), 1)
        self.assertIsNone(audit["invalidation_reason"])
        self.assertTrue(audit["safe_invalidation_evidence"])
        self.assertEqual(audit["candidate_path_count_bucket"], "1")
        self.assertEqual(audit["raw_candidate_path_count_bucket"], "2_5")
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn("emotional/contextual", rendered)
        self.assertNotIn("preferences/facts", rendered)
        self.assertNotIn("src/main.py", rendered)

    def test_tool_result_dependency_capture_fails_closed_for_deleted_paths_without_leaking_names(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            os.chdir(tmp_path)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "cat ./src/main.py ./src/deleted.py",
                            }
                        ],
                    }
                ],
            }

            audit = cache_module.cache_file_dependency_audit(body)

        self.assertEqual(audit["snapshot_count"], 2)
        self.assertEqual(audit["invalidation_reason"], "dependency-missing")
        self.assertEqual(audit["dependency_capture_reason"], "complete")
        self.assertFalse(audit["safe_invalidation_evidence"])
        self.assertFalse(audit["paths_included"])
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn("deleted.py", rendered)
        self.assertNotIn("main.py", rendered)

    def test_file_dependency_fingerprint_metadata_omits_raw_paths_and_records_blockers(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('hello')\n", encoding="utf-8")
            os.chdir(tmp_path)
            body = {"messages": [{"role": "user", "content": "Read ./src/example.py"}]}
            snapshots = cache_module.cache_file_dependency_snapshots(body)
            audit = cache_module.cache_file_dependency_audit(body)

        meta = cache_module.attach_file_dependency_cache_meta(
            {"status": "skipped", "reason": "tools-disabled"},
            snapshots=snapshots,
            audit=audit,
            blocker_reasons=["tool-call-cache-disabled"],
        )
        rendered = json.dumps(meta, sort_keys=True)

        self.assertEqual(meta["file_dependency_count"], 1)
        self.assertEqual(meta["file_dependency_count_bucket"], "1")
        self.assertTrue(meta["file_dependency_fingerprint_available"])
        self.assertTrue(meta["file_dependency_fingerprint_sha256"].startswith("sha256:"))
        self.assertTrue(meta["safe_invalidation_evidence"])
        self.assertIn("tool-call-cache-disabled", meta["cache_replay_blocker_reasons"])
        self.assertFalse(meta["file_dependency_audit"]["paths_included"])
        self.assertFalse(meta["file_dependency_fingerprint"]["paths_included"])
        self.assertFalse(meta["file_dependency_fingerprint"]["path_hashes_included"])
        self.assertNotIn("src/example.py", rendered)
        self.assertNotIn(str(watched), rendered)

    def test_exact_cache_entry_is_invalidated_when_watched_file_changes(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('old')\n", encoding="utf-8")
            body = {"messages": [{"role": "user", "content": "Read src/example.py"}]}
            os.chdir(tmp_path)
            deps = cache_module.cache_file_dependency_snapshots(body)
            store = Store(str(tmp_path / "cache.sqlite3"))
            try:
                store.set_cache("cache-key", "model", 10, {"content": "old"}, file_deps=deps)
                self.assertEqual(store.get_cache("cache-key"), {"content": "old"})
                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertEqual(cached, {"content": "old"})
                self.assertIsNone(reason)

                watched.write_text("print('newer content')\n", encoding="utf-8")

                audit = store.cache_file_dependency_audit("cache-key")
                self.assertEqual(audit["changed_path_count"], 1)
                self.assertEqual(audit["invalidation_reason"], "dependency-changed")
                self.assertFalse(audit["paths_included"])

                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertIsNone(cached)
                self.assertEqual(reason, "dependency-changed")
                cache_row = store.conn.execute(
                    "select 1 from cache where cache_key = ?",
                    ("cache-key",),
                ).fetchone()
                self.assertIsNone(cache_row)
            finally:
                store.conn.close()

    def test_exact_cache_entry_is_invalidated_when_watched_file_is_deleted(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('old')\n", encoding="utf-8")
            body = {"messages": [{"role": "user", "content": "Read src/example.py"}]}
            os.chdir(tmp_path)
            deps = cache_module.cache_file_dependency_snapshots(body)
            store = Store(str(tmp_path / "cache.sqlite3"))
            try:
                store.set_cache("cache-key", "model", 10, {"content": "old"}, file_deps=deps)
                watched.unlink()

                audit = store.cache_file_dependency_audit("cache-key")
                self.assertEqual(audit["deleted_path_count"], 1)
                self.assertEqual(audit["invalidation_reason"], "dependency-deleted")

                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertIsNone(cached)
                self.assertEqual(reason, "dependency-deleted")
            finally:
                store.conn.close()

    def test_exact_cache_entry_without_dependency_evidence_expires_by_ttl(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = Store(str(tmp_path / "cache.sqlite3"))
            try:
                store.set_cache("cache-key", "model", 10, {"content": "old"}, file_deps=[], ttl_seconds=60)
                row = store.conn.execute(
                    "select expires_at from cache where cache_key = ?",
                    ("cache-key",),
                ).fetchone()
                self.assertIsNotNone(row["expires_at"])

                store.conn.execute(
                    "update cache set expires_at = ? where cache_key = ?",
                    ("2000-01-01T00:00:00+00:00", "cache-key"),
                )
                store.conn.commit()

                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertIsNone(cached)
                self.assertEqual(reason, "ttl-expired")
                cache_row = store.conn.execute(
                    "select 1 from cache where cache_key = ?",
                    ("cache-key",),
                ).fetchone()
                self.assertIsNone(cache_row)
            finally:
                store.conn.close()

    def test_exact_cache_entry_with_dependency_evidence_does_not_get_ttl_fallback(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('stable')\n", encoding="utf-8")
            body = {"messages": [{"role": "user", "content": "Read src/example.py"}]}
            os.chdir(tmp_path)
            deps = cache_module.cache_file_dependency_snapshots(body)
            store = Store(str(tmp_path / "cache.sqlite3"))
            try:
                store.set_cache("cache-key", "model", 10, {"content": "stable"}, file_deps=deps, ttl_seconds=60)
                row = store.conn.execute(
                    "select expires_at from cache where cache_key = ?",
                    ("cache-key",),
                ).fetchone()
                self.assertIsNone(row["expires_at"])
                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertEqual(cached, {"content": "stable"})
                self.assertIsNone(reason)
            finally:
                store.conn.close()

    def test_exact_cache_entry_is_invalidated_when_missing_file_appears(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            body = {"messages": [{"role": "user", "content": "Read ./missing.txt"}]}
            os.chdir(tmp_path)
            deps = cache_module.cache_file_dependency_snapshots(body)
            store = Store(str(tmp_path / "cache.sqlite3"))
            try:
                store.set_cache("cache-key", "model", 10, {"content": "missing"}, file_deps=deps)
                self.assertEqual(store.get_cache("cache-key"), {"content": "missing"})

                (tmp_path / "missing.txt").write_text("created\n", encoding="utf-8")

                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertIsNone(cached)
                self.assertEqual(reason, "dependency-changed")
            finally:
                store.conn.close()

    def _session_cache_replay_rules(self, pattern_hash, *, canary_fraction="1.0"):
        return f"""
exact_cache:
  enabled: true
  cache_tool_calls: false
pattern_rules:
  - id: reviewed-session-cache-replay
    enabled: true
    policy_source: managed-recommended
    candidate_id: session-cache-replay-candidate
    conditions:
      pattern_hashes:
        - {pattern_hash}
      source_surface: anthropic_messages
      app_family: claude_code
      category: tool-result
      workflow_phase: tool-result
      has_tools: true
      stream: false
    rollout:
      schema: tokenclaw.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: {canary_fraction}
      canary_salt: session-cache-replay-test
      canary_unit: request_fingerprint
    action:
      type: exact_cache_pattern
      allow_tool_calls: true
      safe_invalidation: true
      scope: session
"""

    def _session_cache_replay_features(self, pattern_hash):
        return {
            "pattern_hashes": [pattern_hash],
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "category": "tool-result",
            "workflow_phase": "tool-result",
            "text_bucket": "2k_8k_chars",
            "token_bucket": "1k_4k_tokens",
            "has_tools": True,
            "stream": False,
            "request_fingerprint": "fixture-cache-replay-request",
            "raw_pattern_strings_included": False,
        }

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

    def test_session_cache_replay_canary_allows_stable_dependency_hit(self):
        pattern_hash = "sha256:" + "1" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                self._session_cache_replay_rules(pattern_hash),
                encoding="utf-8",
            )
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('stable')\n", encoding="utf-8")
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            body = {"messages": [{"role": "user", "content": "Read src/example.py"}]}
            try:
                can_exact, can_semantic, meta = manual.cache_lookup_meta(
                    has_tool_blocks=True,
                    pattern_features=self._session_cache_replay_features(pattern_hash),
                    store_obj=store,
                )
                self.assertTrue(can_exact)
                self.assertFalse(can_semantic)
                scope, scope_id, _rule = manual.cache_replay_scope_for_meta(meta, "session-a")
                other_scope, other_scope_id, _ = manual.cache_replay_scope_for_meta(meta, "session-b")
                key = manual.cache_key_for(body, "/v1/messages", replay_scope=scope, replay_scope_id=scope_id)
                other_key = manual.cache_key_for(body, "/v1/messages", replay_scope=other_scope, replay_scope_id=other_scope_id)
                self.assertNotEqual(key, other_key)

                deps = manual.cache_file_dependency_snapshots(body)
                store.set_cache(key, "claude-sonnet-4-6", 20, {"content": [{"type": "text", "text": "cached"}]}, file_deps=deps)
                allowed, replay_meta = manual.cache_replay_canary_decision(
                    cache_meta=meta,
                    dependency_audit=store.cache_file_dependency_audit(key),
                    session_id="session-a",
                )
                self.assertTrue(allowed, replay_meta)
                cached, invalidated_reason = store.get_cache_with_reason(key)
                hit_meta = manual.cache_hit_decision_meta(
                    "exact-match",
                    hit_type="exact",
                    exact_enabled=can_exact,
                    semantic_enabled=can_semantic,
                    lookup_meta={**meta, "cache_replay_canary": replay_meta},
                    estimated_saved_cost_usd=0.001,
                )

                self.assertIsNone(invalidated_reason)
                self.assertEqual(cached["content"][0]["text"], "cached")
                self.assertEqual(hit_meta["status"], "hit")
                self.assertEqual(hit_meta["cache_replay_canary"]["status"], "applied")
                self.assertEqual(hit_meta["cache_replay_canary"]["reason"], "dependency-stable")
                self.assertEqual(hit_meta["cache_replay_canary"]["canary_cohort"], "canary_applied")
                self.assertFalse(hit_meta["cache_replay_canary"]["dependency_audit"]["paths_included"])
                event = manual.build_cache_replay_lifecycle_feedback(
                    cache_meta=hit_meta,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    status_code=200,
                    latency_ms=7,
                    retry_count=0,
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.001,
                    category="tool-result",
                    stream=False,
                )
                self.assertIsNotNone(event)
                assert_managed_egress_safe(event)
                self.assertEqual(event["schema"], "tokenclaw.cache_replay_lifecycle_feedback.v1")
                self.assertEqual(event["cohort"], "replayed")
                self.assertEqual(event["cache_decision_status"], "hit")
                self.assertEqual(event["estimated_saved_cost_usd"], 0.001)
                self.assertFalse(event["privacy"]["cache_keys_included"])
                self.assertFalse(event["privacy"]["file_paths_included"])
                self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(self._keys_in(event)))
            finally:
                store.conn.close()

    def test_session_cache_replay_canary_blocks_changed_dependency(self):
        pattern_hash = "sha256:" + "2" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                self._session_cache_replay_rules(pattern_hash),
                encoding="utf-8",
            )
            watched = tmp_path / "src" / "example.py"
            watched.parent.mkdir()
            watched.write_text("print('old')\n", encoding="utf-8")
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            body = {"messages": [{"role": "user", "content": "Read src/example.py"}]}
            try:
                can_exact, _can_semantic, meta = manual.cache_lookup_meta(
                    has_tool_blocks=True,
                    pattern_features=self._session_cache_replay_features(pattern_hash),
                    store_obj=store,
                )
                self.assertTrue(can_exact)
                scope, scope_id, _rule = manual.cache_replay_scope_for_meta(meta, "session-a")
                key = manual.cache_key_for(body, "/v1/messages", replay_scope=scope, replay_scope_id=scope_id)
                store.set_cache(
                    key,
                    "claude-sonnet-4-6",
                    20,
                    {"content": [{"type": "text", "text": "stale"}]},
                    file_deps=manual.cache_file_dependency_snapshots(body),
                )
                watched.write_text("print('new')\n", encoding="utf-8")

                allowed, replay_meta = manual.cache_replay_canary_decision(
                    cache_meta=meta,
                    dependency_audit=store.cache_file_dependency_audit(key),
                    session_id="session-a",
                )

                self.assertFalse(allowed)
                self.assertEqual(replay_meta["status"], "invalidated")
                self.assertEqual(replay_meta["reason"], "dependency-changed")
                self.assertEqual(replay_meta["dependency_audit"]["changed_path_count"], 1)
                self.assertFalse(replay_meta["dependency_audit"]["paths_included"])
                event = manual.build_cache_replay_lifecycle_feedback(
                    cache_meta={
                        **meta,
                        "status": replay_meta["status"],
                        "reason": replay_meta["reason"],
                        "invalidated": True,
                        "invalidation_reason": replay_meta["reason"],
                        "cache_replay_canary": replay_meta,
                    },
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    status_code=200,
                    latency_ms=11,
                    retry_count=0,
                    cost_est_usd=0.002,
                    cost_baseline_usd=0.003,
                    category="tool-result",
                    stream=False,
                )
                self.assertEqual(event["cohort"], "invalidated")
                self.assertIn("dependency-changed", event["invalidation_reason_codes"])
                assert_managed_egress_safe(event)
            finally:
                store.conn.close()

    def test_session_cache_replay_canary_blocks_missing_dependency_evidence(self):
        pattern_hash = "sha256:" + "5" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                self._session_cache_replay_rules(pattern_hash),
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            body = {"messages": [{"role": "user", "content": "Summarize the previous tool output."}]}
            try:
                can_exact, _can_semantic, meta = manual.cache_lookup_meta(
                    has_tool_blocks=True,
                    pattern_features=self._session_cache_replay_features(pattern_hash),
                    store_obj=store,
                )
                self.assertTrue(can_exact)
                meta["file_dependency_audit"] = manual.cache_file_dependency_audit(body)
                scope, scope_id, _rule = manual.cache_replay_scope_for_meta(meta, "session-missing-deps")
                key = manual.cache_key_for(body, "/v1/messages", replay_scope=scope, replay_scope_id=scope_id)
                store.set_cache(
                    key,
                    "claude-sonnet-4-6",
                    20,
                    {"content": [{"type": "text", "text": "cached without dependency evidence"}]},
                    file_deps=[],
                )

                allowed, replay_meta = manual.cache_replay_canary_decision(
                    cache_meta=meta,
                    dependency_audit=store.cache_file_dependency_audit(key),
                    session_id="session-missing-deps",
                )

                self.assertFalse(allowed)
                self.assertEqual(replay_meta["status"], "bypassed")
                self.assertEqual(replay_meta["reason"], "file-dependency-missing")
                self.assertFalse(replay_meta["current_dependency_evidence"]["safe_invalidation_evidence"])
                self.assertFalse(replay_meta["current_dependency_evidence"]["paths_included"])
                self.assertNotIn("cached without dependency evidence", json.dumps(replay_meta))
            finally:
                store.conn.close()

    def test_session_cache_replay_canary_blocks_dependency_cap_exceeded(self):
        pattern_hash = "sha256:" + "6" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                self._session_cache_replay_rules(pattern_hash)
                + """
file_watch:
  enabled: true
  max_paths: 1
""",
                encoding="utf-8",
            )
            (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
            (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
            os.chdir(tmp_path)
            capped = importlib.reload(cache_module)
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            body = {"messages": [{"role": "user", "content": "Read ./a.txt and ./b.txt"}]}
            try:
                can_exact, _can_semantic, meta = capped.cache_lookup_meta(
                    has_tool_blocks=True,
                    pattern_features=self._session_cache_replay_features(pattern_hash),
                    store_obj=store,
                )
                self.assertTrue(can_exact)
                current_audit = capped.cache_file_dependency_audit(body)
                self.assertTrue(current_audit["cap_exceeded"])
                self.assertEqual(current_audit["invalidation_reason"], "dependency-cap-exceeded")
                meta["file_dependency_audit"] = current_audit
                scope, scope_id, _rule = capped.cache_replay_scope_for_meta(meta, "session-cap-exceeded")
                key = capped.cache_key_for(body, "/v1/messages", replay_scope=scope, replay_scope_id=scope_id)
                store.set_cache(
                    key,
                    "claude-sonnet-4-6",
                    20,
                    {"content": [{"type": "text", "text": "truncated dependency cache"}]},
                    file_deps=capped.cache_file_dependency_snapshots(body),
                )
                stored_audit = store.cache_file_dependency_audit(key)
                self.assertTrue(stored_audit["safe_invalidation_evidence"])

                allowed, replay_meta = capped.cache_replay_canary_decision(
                    cache_meta=meta,
                    dependency_audit=stored_audit,
                    session_id="session-cap-exceeded",
                )

                self.assertFalse(allowed)
                self.assertEqual(replay_meta["status"], "bypassed")
                self.assertEqual(replay_meta["reason"], "dependency-cap-exceeded")
                self.assertTrue(replay_meta["current_dependency_evidence"]["cap_exceeded"])
                serialized = json.dumps(replay_meta, sort_keys=True)
                self.assertNotIn("a.txt", serialized)
                self.assertNotIn("b.txt", serialized)
                self.assertNotIn("truncated dependency cache", serialized)
            finally:
                store.conn.close()

    def test_session_cache_replay_canary_holdout_forwards_upstream_with_cohort(self):
        pattern_hash = "sha256:" + "3" * 64
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                self._session_cache_replay_rules(pattern_hash, canary_fraction="0.0"),
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)

            can_exact, can_semantic, meta = manual.cache_lookup_meta(
                has_tool_blocks=True,
                pattern_features=self._session_cache_replay_features(pattern_hash),
            )

            self.assertFalse(can_exact)
            self.assertFalse(can_semantic)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "tools-disabled")
            self.assertEqual(meta["canary_cohort"], "canary_holdout")
            self.assertEqual(meta["canary"]["status"], "holdout")
            self.assertEqual(meta["pattern_rules"]["skip_reasons"][-1]["reason"], "canary_holdout")
            holdout_event = manual.build_cache_replay_lifecycle_feedback(
                cache_meta=meta,
                provider="anthropic",
                source_surface="anthropic_messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                status_code=429,
                latency_ms=42,
                retry_count=2,
                cost_est_usd=0.02,
                cost_baseline_usd=0.02,
                category="tool-result",
                stream=False,
            )
            self.assertEqual(holdout_event["cohort"], "holdout")
            self.assertEqual(holdout_event["retry_count"], 2)
            self.assertEqual(holdout_event["status_class"], "client_error")
            assert_managed_egress_safe(holdout_event)

    def test_cache_replay_lifecycle_feedback_covers_safety_stop_metadata_only(self):
        rule = {
            "rule_id": "reviewed-session-cache-replay",
            "candidate_id": "session-cache-replay-candidate",
            "policy_source": "managed-recommended",
            "reason": "local-canary-safety-stop",
            "safety_stop": {
                "reason": "local-canary-safety-stop",
                "decision": "stop",
                "sample_count": 8,
                "error_rate": 0.25,
                "retry_rate": 0.5,
                "pattern_hash": "sha256:" + "4" * 64,
                "path": "/tmp/private.py",
            },
            "canary": {"enabled": True, "selected": False, "cohort": "safety_stopped"},
        }
        event = cache_module.build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "skipped",
                "reason": "local-canary-safety-stop",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [rule]},
            },
            provider="anthropic",
            source_surface="anthropic_messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=500,
            latency_ms=25,
            retry_count=3,
            cost_est_usd=0.03,
            cost_baseline_usd=0.04,
            category="tool-result",
            stream=False,
        )
        self.assertEqual(event["cohort"], "safety_stopped")
        self.assertEqual(event["safety_stop"]["decision"], "stop")
        self.assertNotIn("sha256:" + "4" * 64, json.dumps(event))
        self.assertNotIn("/tmp/private.py", json.dumps(event))
        assert_managed_egress_safe(event)

    def test_cache_replay_lifecycle_feedback_covers_streaming_applied_and_bypassed(self):
        rule = {
            "rule_id": "reviewed-static-streaming-cache",
            "candidate_id": "streaming-static-candidate",
            "policy_source": "managed-recommended",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied", "status": "applied"},
        }
        applied_event = cache_module.build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "miss",
                "reason": "streaming-exact-pattern-miss",
                "pattern_rule": rule,
                "cache_replay_canary": {
                    "status": "applied",
                    "reason": "no-dependency-required",
                    "canary_cohort": "canary_applied",
                },
            },
            provider="anthropic",
            source_surface="anthropic_messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            latency_ms=20,
            retry_count=0,
            cost_est_usd=0.002,
            cost_baseline_usd=0.002,
            category="short-completion",
            stream=True,
        )
        self.assertEqual(applied_event["cohort"], "applied")
        self.assertEqual(applied_event["event_reason"], "cache-miss")
        self.assertTrue(applied_event["stream"])
        assert_managed_egress_safe(applied_event)

        bypassed_event = cache_module.build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "bypassed",
                "reason": "file-dependency-missing",
                "pattern_rule": rule,
                "cache_replay_canary": {
                    "status": "bypassed",
                    "reason": "file-dependency-missing",
                    "canary_cohort": "canary_applied",
                    "dependency_audit": {
                        "safe_invalidation_evidence": False,
                        "invalidation_reason": "file-dependency-missing",
                        "paths_included": False,
                    },
                },
            },
            provider="anthropic",
            source_surface="anthropic_messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            latency_ms=21,
            retry_count=0,
            cost_est_usd=0.002,
            cost_baseline_usd=0.002,
            category="short-completion",
            stream=True,
        )
        self.assertEqual(bypassed_event["cohort"], "bypassed")
        self.assertEqual(bypassed_event["event_reason"], "replay-bypassed")
        self.assertIn("file-dependency-missing", bypassed_event["invalidation_reason_codes"])
        assert_managed_egress_safe(bypassed_event)

    def test_cache_replay_lifecycle_feedback_redacts_raw_like_rule_metadata(self):
        raw_path = "/tmp/private/project/secret.py"
        raw_prompt = "raw prompt must not leave local machine"
        raw_cache_key = "cache-key-user-workspace-secret"
        event = cache_module.build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "hit",
                "reason": "exact-match",
                "policy_source": "managed-recommended",
                "pattern_rule": {
                    "policy_id": raw_cache_key,
                    "rule_id": raw_path,
                    "candidate_id": raw_prompt,
                    "policy_source": "managed-recommended",
                },
                "cache_replay_canary": {
                    "status": "applied",
                    "reason": "dependency-stable",
                    "dependency_audit": {"safe_invalidation_evidence": True, "paths_included": False},
                    "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
                },
            },
            provider="anthropic",
            source_surface="anthropic_messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            latency_ms=5,
            retry_count=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.002,
            category="tool-result",
            stream=False,
        )

        self.assertEqual(event["cohort"], "replayed")
        self.assertTrue(event["policy_id"].startswith("redacted-policy-id-"))
        self.assertTrue(event["rule_id"].startswith("redacted-rule-id-"))
        self.assertTrue(event["candidate_id"].startswith("redacted-candidate-id-"))
        serialized = json.dumps(event, sort_keys=True)
        self.assertNotIn(raw_path, serialized)
        self.assertNotIn(raw_prompt, serialized)
        self.assertNotIn(raw_cache_key, serialized)
        self.assertFalse(event["privacy"]["file_paths_included"])
        self.assertFalse(event["privacy"]["cache_keys_included"])
        self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(self._keys_in(event)))
        assert_managed_egress_safe(event)

    def test_cache_replay_lifecycle_feedback_rejects_raw_egress_fields(self):
        event = {
            "schema": "tokenclaw.cache_replay_lifecycle_feedback.v1",
            "source_surface": "anthropic_messages",
            "cohort": "replayed",
            "prompt": "raw prompt must not leave local machine",
            "body": {"messages": ["raw body"]},
            "file_path": "/tmp/private.py",
        }
        with self.assertRaises(ManagedEgressBlocked) as raised:
            assert_managed_egress_safe(event)
        blocked = {item["key"] for item in raised.exception.violations}
        self.assertIn("prompt", blocked)
        self.assertIn("body", blocked)
        self.assertIn("file_path", blocked)

    def _streaming_static_features(self, pattern_hash):
        return {
            "pattern_hashes": [pattern_hash],
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "category": "short-completion",
            "workflow_phase": "short-completion",
            "text_bucket": "lt_2k_chars",
            "token_bucket": "lt_1k_tokens",
            "has_tools": False,
            "stream": True,
            "cacheability_bucket": "high",
            "static_information_hint": True,
            "time_sensitive_hint": False,
            "user_specific_hint": False,
            "exact_cache_candidate_hint": True,
            "raw_pattern_strings_included": False,
        }

    def _streaming_static_rule(self, pattern_hash, *, canary_fraction=1.0, safe_invalidation=False):
        return cache_module.normalize_cache_pattern_rules([{
            "id": "reviewed-static-streaming-cache",
            "policy_source": "managed-recommended",
            "candidate_id": "streaming-static-candidate",
            "conditions": {
                "pattern_hashes": [pattern_hash],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "short-completion",
                "stream": True,
                "cacheability_bucket": "high",
                "static_information_hint": True,
                "time_sensitive_hint": False,
                "user_specific_hint": False,
            },
            "rollout": {
                "schema": "tokenclaw.pattern_policy_rollout.v1",
                "recommendation_mode": "canary-only",
                "canary_enabled": True,
                "canary_fraction": canary_fraction,
                "canary_salt": "streaming-static-test",
                "canary_unit": "request_fingerprint",
            },
            "action": {
                "type": "exact_cache_pattern",
                "streaming": True,
                "allow_tool_calls": False,
                "safe_invalidation": safe_invalidation,
            },
        }])[0]

    def test_streaming_cache_lookup_allows_default_no_tool_exact_replay(self):
        can_cache, meta = cache_module.streaming_cache_lookup_meta(has_tool_blocks=False)

        self.assertTrue(can_cache)
        self.assertEqual(meta["status"], "miss")
        self.assertEqual(meta["reason"], "streaming-exact-miss")
        self.assertTrue(meta["exact_enabled"])
        self.assertFalse(meta["semantic_enabled"])

        can_tool_cache, tool_meta = cache_module.streaming_cache_lookup_meta(has_tool_blocks=True)

        self.assertFalse(can_tool_cache)
        self.assertEqual(tool_meta["status"], "skipped")
        self.assertEqual(tool_meta["reason"], "streaming-tools-disabled")

    def test_streaming_static_rule_selects_canary_and_records_metadata(self):
        pattern_hash = "sha256:" + "a" * 64
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_static_rule(pattern_hash),)
            can_cache, meta = cache_module.streaming_cache_lookup_meta(
                has_tool_blocks=False,
                pattern_features=self._streaming_static_features(pattern_hash),
            )

            self.assertTrue(can_cache)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "streaming-exact-pattern-miss")
            self.assertEqual(meta["pattern_rule"]["rule_id"], "reviewed-static-streaming-cache")
            self.assertEqual(meta["pattern_rule"]["candidate_id"], "streaming-static-candidate")
            self.assertEqual(meta["pattern_rule"]["replayability_level"], "local-exact-response")
            self.assertEqual(meta["pattern_rule"]["canary"]["status"], "applied")
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_streaming_static_rule_holdout_goes_upstream_with_canary_metadata(self):
        pattern_hash = "sha256:" + "b" * 64
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_static_rule(pattern_hash, canary_fraction=0.0),)
            can_cache, meta = cache_module.streaming_cache_lookup_meta(
                has_tool_blocks=False,
                pattern_features=self._streaming_static_features(pattern_hash),
            )

            self.assertFalse(can_cache)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "canary_holdout")
            self.assertEqual(meta["canary_cohort"], "canary_holdout")
            self.assertEqual(meta["pattern_rule"]["candidate_id"], "streaming-static-candidate")
            self.assertEqual(meta["pattern_rule"]["canary"]["status"], "holdout")
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_streaming_static_rule_blocks_thinking_turns_even_when_pattern_matches(self):
        pattern_hash = "sha256:" + "e" * 64
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_static_rule(pattern_hash),)
            can_cache, meta = cache_module.streaming_cache_lookup_meta(
                has_tool_blocks=False,
                has_thinking_blocks=True,
                pattern_features=self._streaming_static_features(pattern_hash),
            )

            self.assertFalse(can_cache)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "streaming-thinking-disabled")
            self.assertEqual(
                meta["pattern_rules"]["skip_reasons"][-1]["reason"],
                "streaming-thinking-disabled",
            )
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_streaming_static_rule_safety_stop_prevents_replay(self):
        pattern_hash = "sha256:" + "d" * 64
        old_rules = cache_module.CACHE_PATTERN_RULES
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                rule = self._streaming_static_rule(pattern_hash)
                rule["rollout"]["min_outcome_samples"] = 2
                rule["rollout"]["rollback_threshold"] = 0.5
                cache_module.CACHE_PATTERN_RULES = (rule,)
                features = self._streaming_static_features(pattern_hash)

                can_cache, meta = cache_module.streaming_cache_lookup_meta(
                    has_tool_blocks=False,
                    pattern_features=features,
                    store_obj=store,
                )
                self.assertTrue(can_cache)
                self.assertEqual(meta["pattern_rule"]["canary"]["status"], "applied")

                for index in range(2):
                    store.log_call(
                        id=f"failed-streaming-cache-canary-{index}",
                        created_at=f"2026-06-09T00:2{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=1,
                        cache_hit=0,
                        status_code=500,
                        latency_ms=100,
                        input_tokens_est=100,
                        output_tokens_est=0,
                        actual_input_tokens=100,
                        actual_output_tokens=0,
                        cost_est_usd=0.0,
                        cost_baseline_usd=0.0,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({"category": "short-completion"}),
                        cache_json=stable_json(meta),
                        error="upstream failed",
                        request_json=None,
                        response_json=None,
                        session_id="streaming-cache-safety-stop",
                        category="short-completion",
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        retry_count=0,
                        provider="anthropic",
                    )

                stopped, stopped_meta = cache_module.streaming_cache_lookup_meta(
                    has_tool_blocks=False,
                    pattern_features=features,
                    store_obj=store,
                )

                self.assertFalse(stopped)
                self.assertEqual(stopped_meta["status"], "skipped")
                self.assertEqual(stopped_meta["reason"], "local-canary-safety-stop")
                self.assertEqual(stopped_meta["pattern_rule"]["candidate_id"], "streaming-static-candidate")
                self.assertEqual(stopped_meta["pattern_rule"]["safety_stop"]["sample_count"], 2)
            finally:
                store.conn.close()
                cache_module.CACHE_PATTERN_RULES = old_rules

    def test_streaming_static_rule_skips_low_current_user_specific_and_tool_turns(self):
        pattern_hash = "sha256:" + "c" * 64
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_static_rule(pattern_hash),)
            cases = [
                (False, {"cacheability_bucket": "low"}, "cacheability-bucket-mismatch"),
                (False, {"time_sensitive_hint": True}, "time_sensitive_hint-mismatch"),
                (False, {"user_specific_hint": True}, "user_specific_hint-mismatch"),
                (True, {}, "streaming-tools-disabled"),
            ]
            for has_tools_case, overrides, expected_reason in cases:
                features = self._streaming_static_features(pattern_hash)
                features.update(overrides)
                if has_tools_case:
                    features["has_tools"] = True
                can_cache, meta = cache_module.streaming_cache_lookup_meta(
                    has_tool_blocks=has_tools_case,
                    pattern_features=features,
                )
                self.assertFalse(can_cache)
                self.assertEqual(meta["status"], "skipped")
                self.assertEqual(meta["reason"], expected_reason)
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_capture_candidates_is_off_by_default_for_tool_result_body_without_paths(self):
        """Body with no file paths and capture_candidates disabled → file-dependency-missing."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            os.chdir(tmp_path)

            body = {
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit code 0"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
                ]
            }
            snapshots = cache_module.cache_file_dependency_snapshots(body)
            self.assertEqual(snapshots, [])

            audit = cache_module.cache_file_dependency_audit(body)
            self.assertEqual(audit["snapshot_count"], 0)
            self.assertEqual(audit["invalidation_reason"], "file-dependency-missing")
            self.assertFalse(audit["safe_invalidation_evidence"])
            self.assertFalse(audit["paths_included"])
            self.assertNotIn(tmp, json.dumps(audit))

    def test_capture_candidates_provides_workspace_evidence_when_body_has_no_paths(self):
        """With capture_candidates enabled, workspace files become dependency evidence."""
        os.environ["TOKENCLAW_CACHE_CAPTURE_CANDIDATES"] = "1"
        enabled = importlib.reload(cache_module)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
            (tmp_path / "lib.py").write_text("y = 2\n", encoding="utf-8")
            os.chdir(tmp_path)

            body = {
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit code 0"}]},
                ]
            }
            snapshots = enabled.cache_file_dependency_snapshots(body)
            self.assertGreater(len(snapshots), 0)

            audit = enabled.cache_file_dependency_audit(body)
            self.assertGreater(audit["snapshot_count"], 0)
            self.assertIsNone(audit["invalidation_reason"])
            self.assertTrue(audit["safe_invalidation_evidence"])
            self.assertTrue(audit["file_dependency_evidence_available"])
            self.assertFalse(audit["paths_included"])
            self.assertNotIn(tmp, json.dumps(audit))

    def test_capture_candidates_override_collects_workspace_evidence_without_policy_reload(self):
        """OpenAI review-only evidence can snapshot workspace files while default policy stays off."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
            os.chdir(tmp_path)

            body = {
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit code 0"}]},
                ]
            }
            default_audit = cache_module.cache_file_dependency_audit(body)
            snapshots = cache_module.cache_file_dependency_snapshots(body, capture_candidates=True)
            audit = cache_module.cache_file_dependency_audit(body, capture_candidates=True)

        self.assertEqual(default_audit["snapshot_count"], 0)
        self.assertEqual(default_audit["invalidation_reason"], "file-dependency-missing")
        self.assertGreater(len(snapshots), 0)
        self.assertGreater(audit["snapshot_count"], 0)
        self.assertIsNone(audit["invalidation_reason"])
        self.assertTrue(audit["safe_invalidation_evidence"])
        self.assertTrue(audit["file_dependency_evidence_available"])
        self.assertFalse(audit["paths_included"])
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn(str(tmp_path), rendered)
        self.assertNotIn("main.py", rendered)

    def test_capture_candidates_override_fails_closed_for_explicit_deleted_paths(self):
        """Workspace fallback must not hide explicit missing path evidence."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            os.chdir(tmp_path)

            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "cat ./src/main.py ./src/deleted.py",
                            }
                        ],
                    }
                ],
            }
            snapshots = cache_module.cache_file_dependency_snapshots(body, capture_candidates=True)
            audit = cache_module.cache_file_dependency_audit(body, capture_candidates=True)

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(audit["snapshot_count"], 2)
        self.assertEqual(audit["invalidation_reason"], "dependency-missing")
        self.assertFalse(audit["safe_invalidation_evidence"])
        self.assertFalse(audit["file_dependency_evidence_available"])
        self.assertFalse(audit["paths_included"])
        rendered = json.dumps(audit, sort_keys=True)
        self.assertNotIn(str(tmp_path), rendered)
        self.assertNotIn("deleted.py", rendered)

    def test_capture_candidates_dry_run_fixture_shows_transition_to_evidence_present(self):
        """Dry-run fixture: tool-result candidate moves from file-dependency-missing to evidence present."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src.py").write_text("# source\n", encoding="utf-8")
            os.chdir(tmp_path)

            body = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ran test"}]}]}

            # State before: capture_candidates off, no dependency evidence
            audit_before = cache_module.cache_file_dependency_audit(body)
            self.assertEqual(audit_before["invalidation_reason"], "file-dependency-missing")
            self.assertFalse(audit_before["safe_invalidation_evidence"])
            self.assertFalse(audit_before["paths_included"])

            # State after: capture_candidates on, workspace evidence present
            os.environ["TOKENCLAW_CACHE_CAPTURE_CANDIDATES"] = "1"
            enabled = importlib.reload(cache_module)
            audit_after = enabled.cache_file_dependency_audit(body)
            self.assertIsNone(audit_after["invalidation_reason"])
            self.assertTrue(audit_after["safe_invalidation_evidence"])
            self.assertTrue(audit_after["file_dependency_evidence_available"])

            # Privacy: no raw paths in public audit output
            self.assertFalse(audit_after["paths_included"])
            self.assertNotIn(str(tmp_path), json.dumps(audit_after))
            self.assertNotIn("src.py", json.dumps(audit_after))

    def test_stream_cache_payload_round_trips_sse_frames(self):
        frames = [
            b'event: message_start\ndata: {"type":"message_start"}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        payload = cache_module.stream_cache_payload(
            frames,
            provider="anthropic",
            usage={"input_tokens": 12, "output_tokens": 3},
            output_text="hello",
        )

        self.assertTrue(cache_module.is_stream_cache_payload(payload, provider="anthropic"))
        self.assertFalse(cache_module.is_stream_cache_payload(payload, provider="openai"))
        self.assertEqual(cache_module.stream_cache_frames(payload), frames)
        self.assertEqual(payload["usage"]["input_tokens"], 12)
        self.assertEqual(payload["output_text"], "hello")
        self.assertEqual(payload["sse"]["media_type"], "text/event-stream")
        self.assertEqual(payload["sse"]["frame_count"], 2)
        self.assertTrue(payload["sse"]["complete"])

    def test_stream_cache_validation_rejects_malformed_sse_payload_without_raw_data(self):
        malformed = {
            "tokenclaw_cache_type": "sse-stream",
            "version": 1,
            "provider": "anthropic",
            "frames_b64": [base64.b64encode(b"not an sse frame\n\n").decode("ascii")],
        }

        frames, validation = cache_module.validate_stream_cache_payload(malformed, provider="anthropic")

        self.assertEqual(frames, [])
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["reason"], "sse-data-missing")
        self.assertFalse(validation["raw_payload_included"])
        self.assertNotIn("not an sse frame", json.dumps(validation))


if __name__ == "__main__":
    unittest.main()
