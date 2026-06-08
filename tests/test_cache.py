import unittest
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import agentflow_proxy.cache as cache_module
from agentflow_proxy.store import Store


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
