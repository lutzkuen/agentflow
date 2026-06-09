import unittest
import importlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import agentflow_proxy.cache as cache_module
from agentflow_proxy import cli
from agentflow_proxy.policy_bundle import apply_policy_bundle, validate_policy_bundle
from agentflow_proxy.policy_events import recent_policy_events
from agentflow_proxy.store import Store, stable_json


class CacheDecisionMetaTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CACHE",
        "AGENTFLOW_CACHE_TOOL_CALLS",
        "AGENTFLOW_SEMANTIC_CACHE",
        "AGENTFLOW_SEMANTIC_THRESHOLD",
        "AGENTFLOW_CACHE_RULES",
        "AGENTFLOW_PROVIDER",
        "AGENTFLOW_ANTHROPIC_UPSTREAM",
        "AGENTFLOW_OPENAI_UPSTREAM",
        "AGENTFLOW_CACHE_NAMESPACE",
        "AGENTFLOW_CACHE_FILE_WATCH",
        "AGENTFLOW_CACHE_WATCH_ROOT",
        "AGENTFLOW_CACHE_WATCH_MAX_PATHS",
        "AGENTFLOW_PATTERN_CANARY_SAFETY_STOP",
        "AGENTFLOW_PATTERN_CANARY_SAFETY_STOP_WINDOW",
        "AGENTFLOW_POLICY_EVENTS",
        "AGENTFLOW_POLICY_EVENTS_LOG",
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
        os.environ["AGENTFLOW_CACHE_NAMESPACE"] = "env-project"
        os.environ["AGENTFLOW_PROVIDER"] = "openai"
        os.environ["AGENTFLOW_OPENAI_UPSTREAM"] = "https://openai.example"

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
            self.assertEqual(meta["rule_path"], str(config / "cache_rules.yaml"))
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

            os.environ["AGENTFLOW_CACHE_RULES"] = str(cache_rules_path)
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
      schema: agentflow.pattern_policy_rollout.v1
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
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(tmp_path / "policy_events.jsonl")
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
      schema: agentflow.pattern_policy_rollout.v1
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
            store = Store(str(tmp_path / "agentflow.sqlite3"))
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
                                "schema": "agentflow.pattern_canary_decision.v1",
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

    def test_file_dependency_snapshots_can_be_disabled_by_policy(self):
        os.environ["AGENTFLOW_CACHE_FILE_WATCH"] = "0"
        disabled = importlib.reload(cache_module)

        snapshots = disabled.cache_file_dependency_snapshots({
            "messages": [{"role": "user", "content": "Read agentflow_proxy/cache.py"}]
        })

        self.assertEqual(snapshots, [])

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

                cached, reason = store.get_cache_with_reason("cache-key")
                self.assertIsNone(cached)
                self.assertEqual(reason, "file-dependency-changed")
                cache_row = store.conn.execute(
                    "select 1 from cache where cache_key = ?",
                    ("cache-key",),
                ).fetchone()
                self.assertIsNone(cache_row)
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

                self.assertIsNone(store.get_cache("cache-key"))
            finally:
                store.conn.close()

    def test_streaming_cache_lookup_is_exact_only_and_skips_tools_by_default(self):
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


if __name__ == "__main__":
    unittest.main()
