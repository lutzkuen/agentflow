import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.policy_events import recent_policy_events
from agentflow_proxy.store import Store, stable_json


def _thinking_body(
    *thinking_texts: str,
    top_level_thinking: bool = False,
    include_tool_use: bool = False,
    malformed: bool = False,
) -> dict:
    content = []
    for text in thinking_texts:
        content.append({"type": "thinking", "thinking": text})
    if malformed:
        content.append({"type": "thinking", "thinking": {"secret": "raw-malformed-secret"}})
    content.append({"type": "text", "text": "bounded assistant fallback"})
    if include_tool_use:
        content.append({
            "type": "tool_use",
            "id": "raw-tool-use-id-must-not-leak",
            "name": "Read",
            "input": {"file_path": "/workspace/private/secret.py"},
        })
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "assistant", "content": content}],
    }
    if include_tool_use:
        body["messages"].append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "raw-tool-use-id-must-not-leak",
                "content": "raw tool payload secret must not leak",
            }],
        })
    if top_level_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    return body


class CrunchRulesTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CRUNCH",
        "AGENTFLOW_CRUNCH_THRESHOLD_CHARS",
        "AGENTFLOW_PROMPT_CACHE",
        "AGENTFLOW_PROMPT_CACHE_MIN_CHARS",
        "AGENTFLOW_CRUNCH_RULES",
        "AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT",
        "AGENTFLOW_HAIKU_SUMMARY_MODEL",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_REQUEST_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_SUMMARIZED_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_KEEP_RECENT_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SUMMARY_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SOURCE_CHARS",
        "AGENTFLOW_ENHANCED_CRUNCH_MODE",
        "AGENTFLOW_ENHANCED_CRUNCH_MODEL",
        "AGENTFLOW_ENHANCED_CRUNCH_MODEL_FAMILY",
        "AGENTFLOW_ENHANCED_CRUNCH_ENDPOINT_URL",
        "AGENTFLOW_ENHANCED_CRUNCH_MAX_SUMMARY_COST_USD",
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

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(crunch_module)

    def test_default_crunch_policy_reports_bundled_local_default_source(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CRUNCH": "1",
                "AGENTFLOW_CRUNCH_THRESHOLD_CHARS": "24000",
                "AGENTFLOW_PROMPT_CACHE": "1",
                "AGENTFLOW_PROMPT_CACHE_MIN_CHARS": "4096",
                "AGENTFLOW_CRUNCH_RULES": "",
            },
        ):
            manual = importlib.reload(crunch_module)

            _, meta = manual.crunch_body({"model": "claude-sonnet-4-6", "messages": []})

            self.assertTrue(meta["enabled"])
            self.assertEqual(meta["policy_source"], "local-default")
            self.assertTrue(meta["rule_path"].endswith("agentflow_proxy/crunch_rules.yaml"))

    def test_anthropic_thinking_metadata_records_duplicate_local_hashes_privately(self):
        manual = importlib.reload(crunch_module)
        secret = "raw duplicate thinking secret /workspace/private/plan.py "
        thinking = (secret + "alpha beta gamma delta epsilon ") * 120
        body = _thinking_body(thinking, thinking)

        _crunched, meta = manual.crunch_body(
            body,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
        )

        thinking_meta = meta["anthropic_thinking_history"]
        self.assertEqual(thinking_meta["schema"], "agentflow.anthropic_thinking_history_metadata.v1")
        self.assertEqual(thinking_meta["status"], "ready")
        self.assertEqual(thinking_meta["reason"], "ready-for-thinking-compaction-planning")
        self.assertEqual(thinking_meta["thinking_block_count"], 2)
        self.assertEqual(thinking_meta["exact_duplicate_thinking_block_count"], 1)
        self.assertEqual(thinking_meta["near_duplicate_thinking_block_count"], 0)
        self.assertEqual(thinking_meta["unique_local_thinking_block_fingerprint_count"], 1)
        self.assertTrue(thinking_meta["local_thinking_block_fingerprints_included"])
        self.assertEqual(thinking_meta["policy_source"], "local-default")
        self.assertTrue(thinking_meta["rule_path"].endswith("agentflow_proxy/crunch_rules.yaml"))
        self.assertFalse(thinking_meta["raw_thinking_text_included"])
        rendered = json.dumps(thinking_meta, sort_keys=True)
        self.assertNotIn("raw duplicate thinking secret", rendered)
        self.assertNotIn("/workspace/private", rendered)
        self.assertNotIn("alpha beta gamma", rendered)

    def test_anthropic_thinking_metadata_records_non_duplicate_blocks(self):
        manual = importlib.reload(crunch_module)
        body = _thinking_body(
            ("first private reasoning " + "alpha beta gamma delta ") * 120,
            ("second private reasoning " + "red blue green yellow ") * 120,
        )

        _crunched, meta = manual.crunch_body(
            body,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
        )

        thinking_meta = meta["anthropic_thinking_history"]
        self.assertEqual(thinking_meta["status"], "ready")
        self.assertEqual(thinking_meta["thinking_block_count"], 2)
        self.assertEqual(thinking_meta["exact_duplicate_thinking_block_count"], 0)
        self.assertEqual(thinking_meta["near_duplicate_thinking_block_count"], 0)
        self.assertEqual(thinking_meta["unique_local_thinking_block_fingerprint_count"], 2)
        buckets = {item["bucket"]: item["count"] for item in thinking_meta["block_size_bucket_breakdown"]}
        self.assertGreater(sum(buckets.values()), 0)

    def test_anthropic_thinking_metadata_blocks_active_top_level_thinking(self):
        manual = importlib.reload(crunch_module)
        body = _thinking_body(
            ("active thinking secret " + "alpha beta gamma delta ") * 120,
            top_level_thinking=True,
        )

        _crunched, meta = manual.crunch_body(
            body,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
        )

        thinking_meta = meta["anthropic_thinking_history"]
        self.assertEqual(thinking_meta["status"], "blocked")
        self.assertTrue(thinking_meta["top_level_thinking_active"])
        self.assertIn("active-top-level-thinking-request", thinking_meta["blockers"])

    def test_anthropic_thinking_metadata_blocks_adjacent_tool_use_dependency_privately(self):
        manual = importlib.reload(crunch_module)
        body = _thinking_body(
            ("tool dependent thinking secret " + "alpha beta gamma delta ") * 120,
            include_tool_use=True,
        )

        _crunched, meta = manual.crunch_body(
            body,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
        )

        thinking_meta = meta["anthropic_thinking_history"]
        self.assertEqual(thinking_meta["status"], "blocked")
        self.assertEqual(thinking_meta["adjacent_tool_use_dependency_count"], 1)
        self.assertIn("adjacent-tool-use-dependency", thinking_meta["blockers"])
        rendered = json.dumps(thinking_meta, sort_keys=True)
        self.assertNotIn("raw-tool-use-id-must-not-leak", rendered)
        self.assertNotIn("raw tool payload secret", rendered)
        self.assertNotIn("/workspace/private", rendered)

    def test_anthropic_thinking_metadata_blocks_malformed_content_shape(self):
        manual = importlib.reload(crunch_module)
        body = _thinking_body(malformed=True)

        _crunched, meta = manual.crunch_body(
            body,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
        )

        thinking_meta = meta["anthropic_thinking_history"]
        self.assertEqual(thinking_meta["status"], "blocked")
        self.assertGreaterEqual(thinking_meta["unsupported_content_block_shape_count"], 1)
        self.assertIn("unsupported-content-block-shape", thinking_meta["blockers"])
        self.assertNotIn("raw-malformed-secret", json.dumps(thinking_meta, sort_keys=True))

    def test_anthropic_thinking_metadata_mode_blocks_without_raw_body(self):
        manual = importlib.reload(crunch_module)

        thinking_meta = manual.build_anthropic_thinking_history_metadata(
            None,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            category="tool-result",
        )

        self.assertEqual(thinking_meta["status"], "blocked")
        self.assertEqual(thinking_meta["privacy_mode"], "metadata-only")
        self.assertFalse(thinking_meta["body_available"])
        self.assertIn("raw-body-unavailable", thinking_meta["blockers"])
        self.assertFalse(thinking_meta["local_thinking_block_fingerprints_included"])
        self.assertFalse(thinking_meta["raw_messages_included"])

    def test_config_crunch_rules_can_disable_crunch_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text("enabled: false\n", encoding="utf-8")
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            body = {"messages": [{"role": "user", "content": "hello"}]}
            crunched, meta = manual.crunch_body(body)

            self.assertIs(crunched, body)
            self.assertFalse(meta["enabled"])
            self.assertFalse(meta["changed"])
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["rule_path"], str(config / "crunch_rules.yaml"))

    def test_config_crunch_rules_can_change_shortening_threshold_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 10
prompt_cache:
  enabled: true
  min_chars: 4096
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            long_text = "alpha " * 1600
            body = {
                "messages": [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "one"},
                    {"role": "user", "content": "two"},
                    {"role": "assistant", "content": "three"},
                    {"role": "user", "content": "four"},
                ]
            }
            crunched, meta = manual.crunch_body(body)

            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["threshold_chars"], 10)
            self.assertEqual(meta["long_blocks_shortened"], 1)
            self.assertIn("middle of long older text block omitted", crunched["messages"][0]["content"])

    def test_crunch_rule_taxonomy_records_lossless_applied_rule(self):
        manual = importlib.reload(crunch_module)

        _crunched, meta = manual.crunch_body({
            "messages": [{"role": "user", "content": "alpha    beta\n\n\n\n gamma"}],
        })

        applied = {row["rule_id"]: row for row in meta["applied_rules"]}
        self.assertIn("whitespace_normalization", applied)
        whitespace = applied["whitespace_normalization"]
        self.assertEqual(whitespace["rule_group"], "lossless_normalization")
        self.assertEqual(whitespace["lossiness_class"], "lossless")
        self.assertEqual(whitespace["canary_state"], "active")
        self.assertFalse(whitespace["requires_server_decision"])
        rendered = json.dumps(meta, sort_keys=True)
        self.assertNotIn('"raw_prompt"', rendered)

    def test_crunch_rule_taxonomy_tracks_semantic_risk_held_canary(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
crunch:
  enabled: true
  allowed_rules:
    - anthropic_thinking_history_compaction
  anthropic_thinking_history_compaction:
    enabled: true
    canary:
      enabled: true
      canary_fraction: 0.0
      holdout_fraction: 1.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            _crunched, meta = manual.crunch_body({"messages": [{"role": "user", "content": "hello"}]})

            taxonomy = {row["rule_id"]: row for row in meta["rule_taxonomy"]["rules"]}
            semantic = taxonomy["anthropic_thinking_history_compaction"]
            self.assertEqual(semantic["rule_group"], "semantic_compaction")
            self.assertEqual(semantic["lossiness_class"], "semantic_loss_risk")
            self.assertTrue(semantic["requires_canary"])
            self.assertEqual(semantic["canary_state"], "held")

    def test_allowed_rules_blocks_old_text_collapse_with_rule_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
crunch:
  enabled: true
  threshold_chars: 10
  allowed_rules:
    - whitespace_normalization
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            long_text = "alpha " * 1600
            body = {
                "messages": [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "one"},
                    {"role": "user", "content": "two"},
                    {"role": "assistant", "content": "three"},
                    {"role": "user", "content": "four"},
                ]
            }

            crunched, meta = manual.crunch_body(body)

            self.assertEqual(crunched["messages"][0]["content"], long_text.strip())
            self.assertEqual(meta["long_blocks_shortened"], 0)
            skipped = {
                (row["rule_id"], row["reason"]): row
                for row in meta["skipped_rules"]
            }
            blocked = skipped[("old_text_collapse", "rule-not-allowed")]
            self.assertEqual(blocked["rule_group"], "structural_dedup")
            self.assertEqual(blocked["lossiness_class"], "structure_preserving")
            self.assertEqual(blocked["status"], "skipped")

    def test_request_shape_repeated_context_canary_records_applied_and_holdout_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
request_shape_repeated_context_canaries:
  enabled: true
  rules:
    - id: local-repeated-context-crunch-canary-test
      enabled: true
      policy_source: local-manual
      cohort_id: request-shape-crunch:anthropic:messages:tool-result:test
      source_evidence_schema: agentflow.request_shape_crunch_opportunity_dry_run.v1
      source_evidence_schemas:
        - agentflow.request_shape_crunch_opportunity_dry_run.v1
      projected_saved_tokens: 1234
      projected_saved_usd: 0.0037
      staged_at: '2026-06-16T18:00:00+00:00'
      conditions:
        provider_family: anthropic
        source_surface: anthropic_messages
        endpoint: messages
        category: tool-result
        workflow_phase: thinking
        stream: true
        has_tools: true
        text_bucket: 8k_32k_chars
        cache_status: skipped
        routing_status: passthrough
      rollout:
        canary_enabled: true
        canary_fraction: 0.5
        holdout_fraction: 0.5
        canary_salt: local-repeated-context-crunch-canary-test
        canary_unit: request_shape_cohort
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            observed = {}
            for index in range(120):
                tool_payload = ("private tool output variant " + str(index) + " ") * 500
                body = {
                    "model": "claude-sonnet-4-6",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "raw-tool-id-must-not-leak",
                                    "content": tool_payload,
                                }
                            ],
                        }
                    ],
                }
                routing_meta = {
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "text_chars": len(tool_payload),
                    "has_tools": True,
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                }
                _crunched, meta = manual.crunch_body(
                    body,
                    routing_meta=routing_meta,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                )
                lifecycle = meta.get("request_shape_repeated_context_canary")
                if isinstance(lifecycle, dict) and lifecycle.get("status") in {"applied", "holdout"}:
                    observed.setdefault(str(lifecycle["status"]), (lifecycle, meta["request_shape_repeated_context_canaries"]))
                if {"applied", "holdout"} <= set(observed):
                    break

            self.assertIn("applied", observed)
            self.assertIn("holdout", observed)
            applied_lifecycle, applied_meta = observed["applied"]
            holdout_lifecycle, holdout_meta = observed["holdout"]
            self.assertEqual(applied_lifecycle["cohort"], "canary_applied")
            self.assertEqual(holdout_lifecycle["cohort"], "canary_holdout")
            self.assertEqual(applied_lifecycle["policy_id"], "local-repeated-context-crunch-canary-test")
            self.assertEqual(applied_lifecycle["staged_at"], "2026-06-16T18:00:00+00:00")
            self.assertEqual(applied_lifecycle["source_evidence_schema"], "agentflow.request_shape_crunch_opportunity_dry_run.v1")
            self.assertEqual(applied_lifecycle["projected_saved_tokens"], 1234)
            self.assertEqual(applied_meta["applied_count"], 1)
            self.assertEqual(holdout_meta["holdout_count"], 1)
            self.assertEqual(applied_meta["rules"][0]["policy_source"], "local-manual")
            self.assertEqual(applied_meta["rules"][0]["staged_at"], "2026-06-16T18:00:00+00:00")
            rendered = json.dumps({"applied": applied_meta, "holdout": holdout_meta}, sort_keys=True)
            self.assertNotIn("private tool output", rendered)
            self.assertNotIn("raw-tool-id-must-not-leak", rendered)
            self.assertNotIn('"request_fingerprint":', rendered)
            self.assertFalse(applied_meta["privacy"]["raw_prompts_included"])
            self.assertFalse(applied_meta["privacy"]["provider_bodies_included"])
            self.assertFalse(applied_meta["privacy"]["request_fingerprints_included"])

            mismatch_body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hello"}]}
            _mismatch_crunched, mismatch_meta = manual.crunch_body(
                mismatch_body,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )
            self.assertNotIn("request_shape_repeated_context_canary", mismatch_meta)
            self.assertEqual(mismatch_meta["request_shape_repeated_context_canaries"]["reason"], "no-matching-rule")

    def test_enhanced_crunch_provider_reports_configured_without_leaking_endpoint(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
enhanced_crunch_provider:
  mode: customer_sidecar
  profile: old-context-summary
  model_family: haiku
  endpoint_url: http://127.0.0.1:4811/summarize
old_context_summarization:
  enabled: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            meta = manual.enhanced_crunch_provider_public_meta({
                "policy_source": "managed-recommended",
                "profile": "old-context-summary",
                "old_context_summarization": {"enabled": True, "model_hint": "haiku"},
            })
            rendered = json.dumps(meta, sort_keys=True)

            self.assertTrue(meta["configured"])
            self.assertEqual(meta["state"], "configured")
            self.assertEqual(meta["mode"], "customer_sidecar")
            self.assertTrue(meta["endpoint_configured"])
            self.assertFalse(meta["endpoint_url_included"])
            self.assertNotIn("4811", rendered)

    def test_managed_enhanced_summary_hint_falls_back_without_local_provider(self):
        with patch.dict(os.environ, {"AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT": "0"}):
            manual = importlib.reload(crunch_module)
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "SECRET_OLD_CONTEXT " * 40},
                {"role": "assistant", "content": "old answer " * 40},
                {"role": "user", "content": "recent"},
            ],
        }
        managed_profile = {
            "policy_source": "managed-recommended",
            "profile": "old-context-summary",
            "old_context_summarization": {
                "enabled": True,
                "model_hint": "claude-haiku-4-5-20251001",
                "thresholds": {
                    "min_request_chars": 10,
                    "min_summarized_chars": 10,
                    "keep_recent_turns": 1,
                },
            },
        }

        async def fail_fetch(_summary_request):
            raise AssertionError("fallback-not-configured must not call the summary provider")

        summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
            body,
            exact_cache_enabled=False,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            fetch_summary=fail_fetch,
            managed_profile=managed_profile,
        ))

        rendered = json.dumps(meta, sort_keys=True)
        self.assertEqual(summarized, body)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "fallback-not-configured")
        self.assertEqual(meta["enhanced_crunch_state"], "fallback-not-configured")
        self.assertFalse(meta["enhanced_crunch_provider"]["configured"])
        self.assertNotIn("SECRET_OLD_CONTEXT", rendered)

    def test_managed_enhanced_summary_hint_executes_with_local_provider_configured(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
enhanced_crunch_provider:
  mode: local_provider_account
  profile: old-context-summary
old_context_summarization:
  enabled: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": "SECRET_OLD_CONTEXT keep /tmp/local-only.txt " * 40},
                    {"role": "assistant", "content": "old answer " * 40},
                    {"role": "user", "content": "recent"},
                ],
            }
            managed_profile = {
                "policy_source": "managed-recommended",
                "profile": "old-context-summary",
                "old_context_summarization": {
                    "enabled": True,
                    "model_hint": "claude-haiku-4-5-20251001",
                    "policy_id": "managed-summary-policy",
                    "candidate_id": "managed-summary-candidate",
                    "thresholds": {
                        "min_request_chars": 10,
                        "min_summarized_chars": 10,
                        "keep_recent_turns": 1,
                        "max_summary_chars": 80,
                        "max_source_chars": 10000,
                    },
                    "excluded_categories": [],
                    "safety_stop": {"enabled": False},
                },
            }

            async def local_fetch(summary_request):
                request_text = stable_json(summary_request)
                self.assertIn("SECRET_OLD_CONTEXT", request_text)
                return {
                    "summary": "Keep /tmp/local-only.txt. Continue from recent context.",
                    "summary_input_tokens": 200,
                    "summary_output_tokens": 20,
                    "summary_cost_est_usd": 0.0004,
                    "summary_status_code": 200,
                }

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=local_fetch,
                managed_profile=managed_profile,
            ))

            rendered_meta = json.dumps(meta, sort_keys=True)
            self.assertEqual(meta["status"], "applied")
            self.assertEqual(meta["enhanced_crunch_state"], "applied")
            self.assertEqual(meta["policy_source"], "managed-recommended")
            self.assertEqual(meta["rule_id"], "managed-summary-policy")
            self.assertTrue(meta["enhanced_crunch_provider"]["configured"])
            self.assertNotIn("SECRET_OLD_CONTEXT", rendered_meta)
            self.assertNotIn("Keep /tmp/local-only.txt", rendered_meta)
            self.assertIn("Keep /tmp/local-only.txt", stable_json(summarized))

    def test_pattern_rules_shorten_older_repeated_text_from_local_file(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            scaffold = (
                "Reviewed scaffold section.\n"
                + "\n".join(f"stable repeated instruction line {i}" for i in range(80))
            )
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
pattern_rules:
  - id: reviewed-scaffold
    enabled: true
    policy_source: managed-recommended
    candidate_id: candidate-123
    conditions:
      pattern_hashes:
        - {pattern_hash}
      min_repeated_count: 2
      keep_recent_matches: 1
      min_text_chars: 1000
    action:
      type: shorten
      head_chars: 80
      tail_chars: 70
      max_replacement_chars: 260
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": scaffold},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": scaffold},
                ],
            }

            crunched, meta = manual.crunch_body(body)

            self.assertTrue(meta["changed"])
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["pattern_rules_applied"], 1)
            self.assertGreater(meta["pattern_rule_saved_chars"], 1000)
            pattern_meta = meta["pattern_rules"]
            self.assertEqual(pattern_meta["configured_count"], 1)
            self.assertEqual(pattern_meta["applied_count"], 1)
            self.assertEqual(pattern_meta["rules"][0]["rule_id"], "reviewed-scaffold")
            self.assertEqual(pattern_meta["rules"][0]["candidate_id"], "candidate-123")
            self.assertEqual(pattern_meta["rules"][0]["policy_source"], "managed-recommended")
            self.assertEqual(pattern_meta["rules"][0]["matched_hashes"], [pattern_hash])
            self.assertIn("reviewed crunch pattern applied", crunched["messages"][0]["content"])
            self.assertIn("pattern_hash=", crunched["messages"][0]["content"])
            self.assertEqual(crunched["messages"][2]["content"], scaffold)

    def test_managed_pattern_rule_canary_holdout_is_stable(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            scaffold = (
                "Reviewed canary scaffold section.\n"
                + "\n".join(f"stable canary instruction line {i}" for i in range(80))
            )
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
pattern_rules:
  - id: reviewed-canary
    enabled: true
    policy_source: managed-recommended
    candidate_id: candidate-canary
    conditions:
      pattern_hashes:
        - {pattern_hash}
      min_repeated_count: 1
      keep_recent_matches: 0
      min_text_chars: 1000
    rollout:
      schema: agentflow.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: 0.10
      canary_salt: sha256:0000000000000000000000000000000000000000000000000000000000000000
      canary_unit: request_fingerprint
    action:
      type: shorten
      head_chars: 80
      tail_chars: 70
      max_replacement_chars: 260
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": scaffold},
                ],
            }

            crunched_1, meta_1 = manual.crunch_body(body)
            crunched_2, meta_2 = manual.crunch_body(body)

            self.assertEqual(crunched_1, body)
            self.assertEqual(crunched_2, body)
            self.assertFalse(meta_1["changed"])
            self.assertEqual(meta_1["pattern_rules_applied"], 0)
            rule_meta = meta_1["pattern_rules"]["rules"][0]
            self.assertEqual(rule_meta["candidate_id"], "candidate-canary")
            self.assertEqual(rule_meta["holdout_count"], 1)
            self.assertEqual(rule_meta["canary"]["status"], "holdout")
            self.assertEqual(rule_meta["canary"], meta_2["pattern_rules"]["rules"][0]["canary"])
            reasons = {(item["reason"], item.get("pattern_hash")) for item in rule_meta["skip_reasons"]}
            self.assertIn(("canary_holdout", pattern_hash), reasons)

            (config / "crunch_rules.yaml").write_text("enabled: true\npattern_rules: []\n", encoding="utf-8")
            no_rule = importlib.reload(crunch_module)
            _, default_meta = no_rule.crunch_body(body)
            self.assertEqual(default_meta["pattern_rules"]["configured_count"], 0)

    def test_managed_pattern_rule_safety_stop_bypasses_failed_canary(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(tmp_path / "policy_events.jsonl")
            scaffold = (
                "Reviewed risky canary scaffold section.\n"
                + "\n".join(f"risky canary instruction line {i}" for i in range(80))
            )
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
pattern_rules:
  - id: reviewed-safety-stop
    enabled: true
    policy_source: managed-recommended
    candidate_id: candidate-safety-stop
    conditions:
      pattern_hashes:
        - {pattern_hash}
      min_repeated_count: 1
      keep_recent_matches: 0
      min_text_chars: 1000
    rollout:
      schema: agentflow.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: 1.0
      canary_salt: local-safety-stop-test
      canary_unit: request_fingerprint
      min_outcome_samples: 2
      rollback_threshold: 0.5
    action:
      type: shorten
      head_chars: 80
      tail_chars: 70
      max_replacement_chars: 260
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": scaffold}],
            }

            crunched, healthy_meta = manual.crunch_body(body, store_obj=store)
            self.assertTrue(healthy_meta["changed"])
            self.assertIn("reviewed crunch pattern applied", crunched["messages"][0]["content"])

            for index in range(2):
                store.log_call(
                    id=f"failed-canary-{index}",
                    created_at=f"2026-06-09T00:0{index}:00+00:00",
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
                    crunch_json=stable_json({
                        "changed": True,
                        "policy_source": "managed-recommended",
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "rules": [
                                {
                                    "rule_id": "reviewed-safety-stop",
                                    "candidate_id": "candidate-safety-stop",
                                    "policy_source": "managed-recommended",
                                    "matched_hashes": [pattern_hash],
                                    "applied_count": 1,
                                    "saved_chars": 1200,
                                    "canary": {
                                        "schema": "agentflow.pattern_canary_decision.v1",
                                        "enabled": True,
                                        "selected": True,
                                        "status": "applied",
                                        "cohort": "canary_applied",
                                    },
                                }
                            ],
                        },
                    }),
                    routing_json=stable_json({"category": "chat"}),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    error="upstream failed",
                    request_json=None,
                    response_json=None,
                    session_id="safety-stop",
                    category="chat",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    provider="anthropic",
                )

            stopped, stopped_meta = manual.crunch_body(body, store_obj=store)
            rule_meta = stopped_meta["pattern_rules"]["rules"][0]
            self.assertEqual(stopped, body)
            self.assertFalse(stopped_meta["changed"])
            self.assertEqual(rule_meta["status"], "bypass")
            self.assertEqual(rule_meta["reason"], "local-canary-safety-stop")
            self.assertEqual(rule_meta["safety_stop"]["sample_count"], 2)
            self.assertEqual(rule_meta["safety_stop"]["error_count"], 2)
            self.assertEqual(rule_meta["safety_stop"]["regression_rate"], 1.0)
            self.assertIn("local-canary-safety-stop", json.dumps(stopped_meta))
            events = recent_policy_events(limit=5)["events"]
            self.assertEqual(events[0]["action"], "pattern-canary-safety-stop")
            self.assertEqual(events[0]["details"]["rule_id"], "reviewed-safety-stop")

    def test_pattern_rules_skip_tool_bearing_payloads_with_reason(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            text = " ".join("unsafe repeated scaffold" for _ in range(80))
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(text))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
pattern_rules:
  - id: reviewed-tool-skip
    conditions:
      pattern_hash: {pattern_hash}
      min_repeated_count: 1
    action:
      type: omit
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "tool_result", "tool_use_id": "tool-1", "content": "stateful output"},
                        ],
                    }
                ]
            }

            crunched, meta = manual.crunch_body(body)

            self.assertFalse(meta["changed"])
            self.assertEqual(crunched, body)
            reasons = {(item["rule_id"], item["reason"]) for item in meta["pattern_rules"]["skip_reasons"]}
            self.assertIn(("reviewed-tool-skip", "unsafe-tool-or-action-payload"), reasons)

    def test_pattern_rules_pass_through_non_matching_payloads(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
pattern_rules:
  - id: reviewed-non-match
    conditions:
      pattern_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
      min_repeated_count: 1
    action:
      type: omit
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {"messages": [{"role": "user", "content": "ordinary request"}]}

            crunched, meta = manual.crunch_body(body)

            self.assertFalse(meta["changed"])
            self.assertEqual(crunched, body)
            self.assertEqual(meta["pattern_rules"]["applied_count"], 0)
            self.assertIn(
                {"rule_id": "reviewed-non-match", "reason": "min-repeated-count-not-met", "count": 1},
                meta["pattern_rules"]["skip_reasons"],
            )

    def test_default_pattern_rules_are_inert(self):
        manual = importlib.reload(crunch_module)
        body = {"messages": [{"role": "user", "content": "hello"}]}

        crunched, meta = manual.crunch_body(body)

        self.assertEqual(crunched, body)
        self.assertEqual(meta["policy_source"], "local-default")
        self.assertEqual(meta["pattern_rules"]["configured_count"], 0)
        self.assertEqual(meta["pattern_rules"]["applied_count"], 0)

    def _write_instruction_dedup_policy(self, config: Path, *, fraction: float = 1.0, holdout: float = 0.0) -> None:
        (config / "crunch_rules.yaml").write_text(
            f"""
enabled: true
instruction_section_deduplication:
  enabled: true
  policy_source: local-manual
  min_section_chars: 80
  min_repeated_count: 2
  keep_recent_sections: 1
  max_replacements: 4
  replacement_notice: "[repeated instruction section omitted by AgentFlow test]"
  canary:
    enabled: true
    fraction: {fraction}
    holdout_fraction: {holdout}
    salt: instruction-dedup-test
    unit: instruction_section_fingerprint
  safety_stop:
    enabled: false
""",
            encoding="utf-8",
        )

    def _instruction_body(self, section: str) -> dict[str, object]:
        return {
            "model": "claude-sonnet-4-6",
            "system": [
                {"type": "text", "text": section},
                {"type": "text", "text": section},
            ],
            "messages": [{"role": "user", "content": "Proceed without repeating instructions."}],
        }

    def test_instruction_section_dedup_default_policy_is_noop(self):
        manual = importlib.reload(crunch_module)
        section = ("Default disabled instruction section. " + "Keep this local-only instruction. " * 8).strip()
        body = self._instruction_body(section)

        crunched, meta = manual.crunch_body(body)

        self.assertEqual(crunched, body)
        dedup = meta["instruction_section_deduplication"]
        self.assertFalse(dedup["enabled"])
        self.assertEqual(dedup["status"], "skipped")
        self.assertEqual(dedup["reason"], "disabled")

    def test_instruction_section_dedup_canary_applied_and_holdout_are_deterministic(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            self._write_instruction_dedup_policy(config, fraction=0.5, holdout=0.5)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            observed = {}
            for index in range(80):
                variant = chr(ord("a") + (index % 26)) * (1 + index // 26)
                section = (
                    f"Canary instruction section variant {variant}. "
                    + "Follow these stable coding-agent operating rules without exposing private text. " * 5
                ).strip()
                body = self._instruction_body(section)
                crunched, meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")
                dedup = meta["instruction_section_deduplication"]
                observed.setdefault(dedup["status"], (body, crunched, dedup))
                if {"applied", "holdout"}.issubset(observed):
                    break

            self.assertIn("applied", observed)
            self.assertIn("holdout", observed)
            applied_body, applied_crunched, applied = observed["applied"]
            holdout_body, holdout_crunched, holdout_meta = observed["holdout"]
            self.assertNotEqual(applied_crunched, applied_body)
            self.assertIn("AgentFlow test", applied_crunched["system"][0]["text"])
            self.assertEqual(applied_crunched["system"][1]["text"], applied_body["system"][1]["text"])
            self.assertEqual(applied["applied_count"], 1)
            self.assertEqual(applied["canary"]["cohort"], "canary_applied")
            self.assertGreater(applied["tokens_saved_est"], 0)
            self.assertEqual(holdout_crunched, holdout_body)
            self.assertEqual(holdout_meta["holdout_count"], 1)
            self.assertEqual(holdout_meta["rules"][0]["canary"]["cohort"], "canary_holdout")
            rendered = json.dumps(applied, sort_keys=True)
            self.assertNotIn("Follow these stable", rendered)
            self.assertNotIn("sha256:", rendered)

            second_crunched, second_meta = manual.crunch_body(
                holdout_body,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )
            self.assertEqual(second_crunched, holdout_body)
            self.assertEqual(second_meta["instruction_section_deduplication"]["rules"][0]["canary"], holdout_meta["rules"][0]["canary"])

    def test_instruction_section_dedup_applies_openai_and_codex_instruction_blocks(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            self._write_instruction_dedup_policy(config, fraction=1.0, holdout=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            section = ("OpenAI/Codex instruction section. " + "Keep deterministic local instructions intact in the newest copy. " * 5).strip()
            openai_body = {
                "model": "gpt-5.4",
                "instructions": section,
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": section}]},
                    {"role": "user", "content": [{"type": "input_text", "text": "Do the task."}]},
                ],
            }
            codex_body = {
                "model": "gpt-5-codex",
                "instructions": section,
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": section}]},
                    {"role": "user", "content": "start turn"},
                ],
            }

            openai_crunched, openai_meta = manual.crunch_body(
                openai_body,
                provider="openai",
                source_surface="openai_responses",
                endpoint="responses",
            )
            codex_crunched, codex_meta = manual.crunch_body(
                codex_body,
                provider="openai",
                source_surface="codex_turn",
                endpoint="app_server",
            )

            self.assertEqual(openai_meta["instruction_section_deduplication"]["status"], "applied")
            self.assertIn("AgentFlow test", openai_crunched["instructions"])
            self.assertEqual(openai_crunched["input"][0]["content"][0]["text"], section)
            self.assertEqual(codex_meta["instruction_section_deduplication"]["status"], "applied")
            self.assertEqual(codex_meta["instruction_section_deduplication"]["source_surface"], "codex_turn")
            self.assertIn("AgentFlow test", codex_crunched["instructions"])
            self.assertEqual(codex_crunched["input"][0]["content"][0]["text"], section)

    def test_instruction_section_dedup_preserves_tool_and_thinking_payloads(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            self._write_instruction_dedup_policy(config, fraction=1.0, holdout=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            section = ("Risky instruction section. " + "Do not mutate protocol bearing requests. " * 6).strip()
            tool_body = self._instruction_body(section)
            tool_body["messages"].append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "read", "input": {"path": "file.py"}}],
            })
            thinking_body = self._instruction_body(section)
            thinking_body["messages"].append({
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "private chain state"}],
            })

            tool_crunched, tool_meta = manual.crunch_body(tool_body)
            thinking_crunched, thinking_meta = manual.crunch_body(thinking_body)

            self.assertEqual(tool_crunched, tool_body)
            self.assertEqual(tool_meta["instruction_section_deduplication"]["reason"], "tool-protocol-risk")
            self.assertEqual(thinking_crunched, thinking_body)
            self.assertEqual(thinking_meta["instruction_section_deduplication"]["reason"], "thinking-content-risk")

    def test_instruction_section_dedup_invalid_canary_policy_fails_closed_without_raw_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            section = ("Invalid canary instruction section. " + "Keep local instruction text private. " * 8).strip()
            fingerprint = crunch_module._instruction_hash_text(crunch_module._normalize_instruction_text(section))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
instruction_section_deduplication:
  enabled: true
  policy_source: local-manual
  min_section_chars: 80
  min_repeated_count: 2
  keep_recent_sections: 1
  max_replacements: 4
  replacement_notice: "raw replacement notice secret must not leak"
  canary:
    enabled: true
    fraction: 0.8
    holdout_fraction: 0.4
    salt: raw canary salt secret must not leak
  safety_stop:
    enabled: false
  rules:
    - id: raw-runtime-rule-secret
      candidate_id: raw-runtime-candidate-secret
      instruction_section_fingerprints:
        - {fingerprint}
      enabled: true
      max_replacements: 2
      canary:
        enabled: true
        fraction: 0.8
        holdout_fraction: 0.4
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = self._instruction_body(section)

            crunched, meta = manual.crunch_body(body)

            self.assertEqual(crunched, body)
            dedup = meta["instruction_section_deduplication"]
            self.assertEqual(dedup["status"], "skipped")
            self.assertEqual(dedup["reason"], "invalid-canary-configuration")
            self.assertIn("invalid-canary-configuration", dedup["reason_codes"])
            self.assertFalse(dedup["rules"][0]["canary"]["valid"])
            self.assertIn("invalid-canary-fraction-sum", dedup["rules"][0]["canary"]["validation_errors"])
            self.assertEqual(dedup["rules"][0]["rule_id"], "instruction-section-dedup-policy")
            self.assertTrue(dedup["rules"][0]["candidate_id"].startswith("instruction-dedup-candidate:"))
            self.assertFalse(dedup["rule_path_included"])
            rendered = json.dumps(dedup, sort_keys=True)
            for forbidden in (
                "raw replacement notice secret must not leak",
                "raw canary salt secret must not leak",
                "raw-runtime-rule-secret",
                "raw-runtime-candidate-secret",
                "Invalid canary instruction section",
            ):
                self.assertNotIn(forbidden, rendered)

    def test_instruction_section_dedup_coordinator_suppression_prevents_mutation(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            self._write_instruction_dedup_policy(config, fraction=1.0, holdout=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            section = ("Coordinator instruction section. " + "This would normally be removed from the older copy. " * 6).strip()
            body = self._instruction_body(section)
            routing_meta = {
                "category": "chat",
                "workflow_phase": "planning",
                "optimization_coordinator": {
                    "selected_family": "cache_replay",
                    "reason_codes": ["selected-cache-replay"],
                    "suppressed_families": [
                        {"family": "instruction_section_deduplication", "reason_codes": ["conflicts-with-coordinator-selection"]}
                    ],
                },
            }

            crunched, meta = manual.crunch_body(body, routing_meta=routing_meta)

            dedup = meta["instruction_section_deduplication"]
            self.assertEqual(crunched, body)
            self.assertEqual(dedup["status"], "suppressed")
            self.assertEqual(dedup["reason"], "coordinator-conflict")
            self.assertIn("coordinator-conflict", dedup["reason_codes"])

    def test_repeated_provider_scaffolding_compacts_anthropic_messages_under_reviewed_rule(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            scaffold = (
                "Reviewed provider scaffold.\n"
                + "\n".join(f"stable instruction line {i}" for i in range(24))
            )
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
repeated_provider_scaffolding:
  enabled: true
  min_request_chars: 1
  min_section_chars: 100
  keep_recent_messages: 1
  keep_recent_matches: 0
  rules:
    - id: reviewed-provider-scaffold
      enabled: true
      policy_source: managed-recommended
      candidate_id: candidate-provider-123
      pattern_hashes:
        - {pattern_hash}
      min_repeated_count: 2
      keep_recent_matches: 0
      max_applications: 4
      rollout:
        canary_enabled: true
        canary_fraction: 1.0
        canary_salt: provider-scaffold-test
        canary_unit: request_fingerprint
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "system": "System instruction must stay verbatim.",
                "messages": [
                    {"role": "user", "content": scaffold + "\n\nOld task details."},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": scaffold + "\n\nNewest task tail must remain."},
                ],
            }

            crunched, meta = manual.crunch_body(body)

            provider_meta = meta["repeated_provider_scaffolding"]
            rendered_meta = json.dumps(provider_meta, sort_keys=True)
            self.assertTrue(meta["changed"])
            self.assertEqual(provider_meta["status"], "applied")
            self.assertEqual(provider_meta["reason"], "repeated-provider-scaffolding-crunched")
            self.assertEqual(provider_meta["applied_count"], 1)
            self.assertGreater(provider_meta["saved_chars"], 100)
            self.assertEqual(provider_meta["rules"][0]["rule_id"], "reviewed-provider-scaffold")
            self.assertEqual(provider_meta["rules"][0]["candidate_id"], "candidate-provider-123")
            self.assertEqual(provider_meta["rules"][0]["policy_source"], "managed-recommended")
            self.assertEqual(provider_meta["rules"][0]["canary"]["cohort"], "canary_applied")
            self.assertNotIn(pattern_hash, rendered_meta)
            self.assertNotIn("stable instruction line", rendered_meta)
            self.assertEqual(crunched["system"], body["system"])
            self.assertIn("repeated provider scaffolding omitted", crunched["messages"][0]["content"])
            self.assertIn("Old task details.", crunched["messages"][0]["content"])
            self.assertEqual(crunched["messages"][2]["content"], body["messages"][2]["content"])

    def test_repeated_provider_scaffolding_preserves_tool_and_thinking_protocol(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            scaffold = "Reviewed tool scaffold.\n" + "\n".join(f"tool-safe line {i}" for i in range(24))
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
repeated_provider_scaffolding:
  enabled: true
  min_request_chars: 1
  min_section_chars: 100
  keep_recent_messages: 1
  keep_recent_matches: 0
  rules:
    - id: reviewed-provider-tool-safe
      candidate_id: candidate-tool-safe
      pattern_hashes: [{pattern_hash}]
      min_repeated_count: 2
      keep_recent_matches: 0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "developer", "content": scaffold + "\n\nDeveloper instruction."},
                    {"role": "user", "content": scaffold + "\n\nOld safe request."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": scaffold + "\n\nPrivate chain state."},
                            {"type": "tool_use", "id": "tool-1", "name": "read", "input": {"path": "file.py"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": scaffold + "\n\nTool result wrapper."},
                            {"type": "tool_result", "tool_use_id": "tool-1", "content": "stateful output"},
                        ],
                    },
                    {"role": "user", "content": scaffold + "\n\nNewest task tail."},
                ]
            }

            crunched, meta = manual.crunch_body(body)

            provider_meta = meta["repeated_provider_scaffolding"]
            self.assertTrue(provider_meta["tool_protocol_preserved"])
            self.assertTrue(provider_meta["thinking_preserved"])
            self.assertEqual(provider_meta["applied_count"], 1)
            self.assertEqual(crunched["messages"][0]["content"], body["messages"][0]["content"])
            self.assertIn("repeated provider scaffolding omitted", crunched["messages"][1]["content"])
            self.assertEqual(crunched["messages"][2]["content"], body["messages"][2]["content"])
            self.assertEqual(crunched["messages"][3]["content"], body["messages"][3]["content"])
            self.assertEqual(crunched["messages"][4]["content"], body["messages"][4]["content"])
            self.assertGreaterEqual(provider_meta["safety_skips"]["system_or_developer_message"], 1)
            self.assertGreaterEqual(provider_meta["safety_skips"]["tool_protocol_payload"], 1)
            self.assertGreaterEqual(provider_meta["safety_skips"]["thinking_payload"], 1)

    def test_repeated_provider_scaffolding_compacts_openai_message_input(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            scaffold = "Reviewed OpenAI scaffold.\n" + "\n".join(f"openai line {i}" for i in range(24))
            pattern_hash = "sha256:" + crunch_module.sha256_text(crunch_module.normalize_text(scaffold))
            (config / "crunch_rules.yaml").write_text(
                f"""
enabled: true
repeated_provider_scaffolding:
  enabled: true
  min_request_chars: 1
  min_section_chars: 100
  keep_recent_messages: 1
  keep_recent_matches: 0
  rules:
    - id: reviewed-openai-provider-scaffold
      policy_source: local-manual
      candidate_id: candidate-openai
      pattern_hash: {pattern_hash}
      min_repeated_count: 2
      keep_recent_matches: 0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "gpt-5.4",
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": scaffold + "\n\nDeveloper stays."}]},
                    {"role": "user", "content": [{"type": "input_text", "text": scaffold + "\n\nOld request."}]},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "ack"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": scaffold + "\n\nNewest OpenAI request."}]},
                ],
            }

            crunched, meta = manual.crunch_body(body)

            provider_meta = meta["repeated_provider_scaffolding"]
            self.assertEqual(provider_meta["applied_count"], 1)
            self.assertIn("repeated provider scaffolding omitted", crunched["input"][1]["content"][0]["text"])
            self.assertIn("Old request.", crunched["input"][1]["content"][0]["text"])
            self.assertEqual(crunched["input"][0], body["input"][0])
            self.assertEqual(crunched["input"][3], body["input"][3])

    def test_repeated_provider_scaffolding_is_disabled_by_default_and_ignores_codex_input(self):
        manual = importlib.reload(crunch_module)
        scaffold = "Default inert scaffold.\n" + "\n".join(f"default line {i}" for i in range(24))
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": scaffold + "\n\nOld one."},
                {"role": "user", "content": scaffold + "\n\nNewest two."},
            ],
        }
        codex_like = {"input": scaffold + "\n\n" + scaffold}

        crunched, meta = manual.crunch_body(body)
        codex_crunched, codex_meta = manual.crunch_body(codex_like)

        self.assertEqual(crunched, body)
        self.assertFalse(meta["repeated_provider_scaffolding"]["enabled"])
        self.assertEqual(meta["repeated_provider_scaffolding"]["reason"], "disabled")
        self.assertEqual(codex_crunched, codex_like)
        self.assertEqual(codex_meta["repeated_provider_scaffolding"]["reason"], "disabled")

    def test_terminal_log_boilerplate_simplifies_timestamp_level_prefixes_and_preserves_diagnostics(self):
        manual = importlib.reload(crunch_module)
        lines = [
            "Operator note before captured logs.",
            *[
                f"2026-06-09T22:20:0{idx}Z INFO [pid=4321] worker.py:77 - processed item {idx}"
                for idx in range(6)
            ],
            "2026-06-09T22:20:07Z ERROR [pid=4321] worker.py:77 - failed to open /tmp/important.txt",
            "Traceback (most recent call last):",
            '  File "/tmp/important.py", line 12, in <module>',
            "AssertionError: expected exit code 0",
            "Exit code: 1",
            "Operator note after captured logs.",
        ]
        body = {"messages": [{"role": "user", "content": "\n".join(lines)}]}

        crunched, meta = manual.crunch_body(body)

        text = crunched["messages"][0]["content"]
        self.assertTrue(meta["changed"])
        self.assertGreater(meta["terminal_log_boilerplate_saved_chars"], 0)
        self.assertIn("timestamp_prefix+log_level_prefix+pid_thread_prefix+module_prefix", meta["terminal_log_boilerplate"]["pattern_types"])
        self.assertIn("processed item 0", text)
        self.assertNotIn("2026-06-09T22:20:00Z INFO [pid=4321]", text)
        self.assertIn("2026-06-09T22:20:07Z ERROR [pid=4321] worker.py:77 - failed to open /tmp/important.txt", text)
        self.assertIn('File "/tmp/important.py", line 12, in <module>', text)
        self.assertIn("AssertionError: expected exit code 0", text)
        self.assertIn("Exit code: 1", text)
        self.assertIn("Operator note before captured logs.", text)
        self.assertIn("Operator note after captured logs.", text)
        self.assertTrue(meta["terminal_log_boilerplate"]["error_bearing_lines_preserved"])
        rendered_meta = json.dumps(meta["terminal_log_boilerplate"], sort_keys=True)
        self.assertNotIn("processed item 0", rendered_meta)
        self.assertNotIn("/tmp/important.txt", rendered_meta)

    def test_terminal_log_boilerplate_simplifies_shell_prompts_without_dropping_commands(self):
        manual = importlib.reload(crunch_module)
        terminal = "\n".join([
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ python -m unittest tests.test_alpha",
            "Ran 4 tests in 0.12s",
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ git status --short",
            " M agentflow_proxy/crunch.py",
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ python -m unittest tests.test_beta",
            "Ran 8 tests in 0.18s",
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ echo done",
            "done",
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ python -m unittest tests.test_gamma",
            "Ran 2 tests in 0.03s",
            "lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$ python -m unittest tests.test_delta",
            "Ran 1 test in 0.01s",
        ])
        body = {"messages": [{"role": "user", "content": terminal}]}

        crunched, meta = manual.crunch_body(body)

        text = crunched["messages"][0]["content"]
        self.assertTrue(meta["changed"])
        self.assertIn("shell_prompt_prefix", meta["terminal_log_boilerplate"]["pattern_types"])
        self.assertIn("command: python -m unittest tests.test_alpha", text)
        self.assertIn("command: git status --short", text)
        self.assertIn("command: echo done", text)
        self.assertNotIn("lutz@dev:/very/long/workspace/path/agentflow/subproject/current-run$", text)

    def test_terminal_log_boilerplate_collapses_repeated_shebangs_and_test_markers(self):
        manual = importlib.reload(crunch_module)
        captured = "\n".join(
            ["#!/usr/bin/env bash", "echo one"]
            + ["#!/usr/bin/env bash"] * 12
            + ["echo four"]
            + ["........"] * 20
            + ["FAILED tests/test_app.py::test_real_failure - AssertionError: bad value"]
        )
        body = {"messages": [{"role": "user", "content": captured}]}

        crunched, meta = manual.crunch_body(body)

        text = crunched["messages"][0]["content"]
        terminal_meta = meta["terminal_log_boilerplate"]
        self.assertTrue(meta["changed"])
        self.assertIn("shebang_line", terminal_meta["pattern_types"])
        self.assertIn("test_progress_marker", terminal_meta["pattern_types"])
        self.assertEqual(text.count("#!/usr/bin/env bash"), 1)
        self.assertNotIn("........", text)
        self.assertIn("echo one", text)
        self.assertIn("echo four", text)
        self.assertIn("FAILED tests/test_app.py::test_real_failure - AssertionError: bad value", text)
        self.assertTrue(terminal_meta["error_bearing_lines_preserved"])

    def test_config_crunch_rules_can_disable_terminal_log_boilerplate(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
terminal_log_boilerplate:
  enabled: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            text = "\n".join(f"INFO worker - repeated line {idx}" for idx in range(8))
            body = {"messages": [{"role": "user", "content": text}]}

            crunched, meta = manual.crunch_body(body)

            self.assertEqual(crunched, body)
            self.assertFalse(meta["terminal_log_boilerplate"]["enabled"])
            self.assertEqual(meta["terminal_log_boilerplate"]["reason"], "disabled")

    def test_thinking_near_duplicate_dedup_removes_older_assistant_block(self):
        manual = importlib.reload(crunch_module)
        base_words = [f"token{i}" for i in range(520)]
        newer_words = list(base_words)
        newer_words[200] = "updated-token"
        older_thinking = " ".join(base_words)
        newer_thinking = " ".join(newer_words)
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": older_thinking, "signature": "older-signature"},
                        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": newer_thinking, "signature": "newer-signature"},
                        {"type": "tool_use", "id": "tool-2", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-2", "content": "ok"}]},
            ]
        }

        crunched, meta = manual.crunch_body(body)

        self.assertTrue(meta["changed"])
        self.assertGreater(meta["saved_chars"], 2000)
        self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 1)
        self.assertEqual(crunched["messages"][0]["content"], [
            {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
        ])
        self.assertEqual(crunched["messages"][2]["content"][0]["thinking"], newer_thinking)
        self.assertEqual(crunched["messages"][2]["content"][0]["signature"], "newer-signature")
        self.assertEqual(crunched["messages"][2]["content"][1]["type"], "tool_use")

    def test_thinking_near_duplicate_dedup_preserves_latest_assistant_block(self):
        manual = importlib.reload(crunch_module)
        thinking = " ".join(f"token{i}" for i in range(520))
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": thinking, "signature": "older-signature"},
                        {"type": "text", "text": "older done"},
                    ],
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": thinking, "signature": "latest-signature"},
                        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]},
            ]
        }

        crunched, meta = manual.crunch_body(body)

        latest = crunched["messages"][2]["content"][0]
        self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 1)
        self.assertEqual(latest["type"], "thinking")
        self.assertEqual(latest["thinking"], thinking)
        self.assertEqual(latest["signature"], "latest-signature")
        self.assertTrue(meta["thinking_deduplication"]["skip_latest_assistant"])

    def test_config_crunch_rules_can_disable_thinking_dedup_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
thinking_deduplication:
  enabled: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            thinking = " ".join(f"token{i}" for i in range(520))
            body = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": thinking},
                            {"type": "text", "text": "older"},
                        ],
                    },
                    {"role": "user", "content": "continue"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": thinking},
                            {"type": "text", "text": "latest"},
                        ],
                    },
                ]
            }

            crunched, meta = manual.crunch_body(body)

            self.assertFalse(meta["thinking_deduplication"]["enabled"])
            self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 0)
            self.assertEqual(crunched["messages"][0]["content"][0]["type"], "thinking")

    def test_old_context_summarization_defaults_to_canary_policy(self):
        manual = importlib.reload(crunch_module)
        plan, meta = manual.old_context_summary_plan(
            {"messages": [{"role": "user", "content": "old text " * 10000}]},
            exact_cache_enabled=True,
        )

        self.assertIsNone(plan)
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "not-enough-old-turns")
        self.assertEqual(meta["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["max_summary_cost_usd"], 0.02)
        self.assertEqual(meta["keep_recent_turns"], 4)
        self.assertTrue(meta["block_thinking"])
        self.assertTrue(meta["canary"]["enabled"])
        self.assertEqual(meta["canary"]["fraction"], 0.25)
        self.assertEqual(meta["canary"]["holdout_fraction"], 0.75)
        self.assertEqual(meta["canary"]["salt"], "local-old-context-summary-v1")
        self.assertEqual(meta["canary"]["unit"], "source_hash")
        self.assertTrue(meta["safety_stop"]["enabled"])

    def test_old_context_summarization_legacy_env_can_disable_canary_policy(self):
        with patch.dict(os.environ, {"AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT": "0"}):
            manual = importlib.reload(crunch_module)
            plan, meta = manual.old_context_summary_plan(
                {"messages": [{"role": "user", "content": "old text " * 10000}]},
                exact_cache_enabled=True,
            )

        self.assertIsNone(plan)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "disabled")

    def test_old_context_summarization_does_not_require_exact_cache(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            plan, meta = manual.old_context_summary_plan(
                {
                    "messages": [
                        {"role": "user", "content": "old text " * 20},
                        {"role": "assistant", "content": "recent"},
                    ]
                },
                exact_cache_enabled=False,
            )

            self.assertIsNotNone(plan)
            self.assertTrue(meta["enabled"])
            self.assertEqual(meta["reason"], "eligible")
            self.assertTrue(meta["summary_cache_enabled"])
            self.assertFalse(meta["exact_cache_enabled"])

    def test_old_context_summarization_plans_only_old_non_tool_turns(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 3
  keep_recent_turns: 2
  max_summary_chars: 1000
  max_source_chars: 20000
  block_tool_protocol: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 20},
                    {"role": "assistant", "content": [{"type": "text", "text": "beta " * 20}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file"}]},
                    {"role": "user", "content": "gamma " * 20},
                    {"role": "assistant", "content": "recent assistant"},
                    {"role": "user", "content": "recent user"},
                ]
            }

            plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=True)

            self.assertIsNotNone(plan)
            self.assertEqual(plan["candidate_indexes"], [0, 1, 3])
            self.assertEqual(meta["status"], "planned")
            self.assertEqual(meta["eligible_turns"], 3)
            self.assertIn("claude-haiku-4-5-20251001", plan["summary_request"]["model"])

    def test_maybe_summarize_old_context_uses_cached_summary_without_fetch(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 40},
                    {"role": "assistant", "content": "beta " * 40},
                    {"role": "user", "content": "recent"},
                ]
            }
            plan, _ = manual.old_context_summary_plan(body, exact_cache_enabled=True)
            cache = {plan["cache_key"]: {"summary": "durable facts only"}}

            async def fail_fetch(_request):
                raise AssertionError("fetch should not run on summary cache hit")

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=True,
                get_cached_summary=cache.get,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fail_fetch,
            ))

            self.assertTrue(meta["changed"])
            self.assertEqual(meta["status"], "applied")
            self.assertEqual(meta["reason"], "summary-cache-hit")
            self.assertTrue(meta["summary_cache_hit"])
            self.assertIn("durable facts only", summarized["system"][0]["text"])
            self.assertIn("AgentFlow: old non-tool context summarized", summarized["system"][0]["text"])
            self.assertEqual(summarized["messages"], [{"role": "user", "content": "recent"}])

    def test_maybe_summarize_old_context_fetches_and_caches_summary(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 40},
                    {"role": "assistant", "content": "beta " * 40},
                    {"role": "user", "content": "recent"},
                ]
            }
            cache = {}
            fetch_requests = []

            async def fetch(request):
                fetch_requests.append(request)
                return {
                    "summary": "fetched compact summary",
                    "summary_input_tokens": 100,
                    "summary_output_tokens": 20,
                    "summary_cost_est_usd": 0.0008,
                    "summary_status_code": 200,
                }

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=True,
                get_cached_summary=cache.get,
                set_cached_summary=lambda key, value: cache.__setitem__(key, value),
                fetch_summary=fetch,
            ))

            self.assertEqual(len(fetch_requests), 1)
            self.assertEqual(len(cache), 1)
            self.assertEqual(meta["reason"], "summary-created")
            self.assertEqual(meta["summary_input_tokens"], 100)
            self.assertAlmostEqual(meta["summary_cost_est_usd"], 0.0008)
            self.assertIn("fetched compact summary", summarized["system"][0]["text"])

    def test_old_context_summary_preserves_existing_system_as_system_context(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "system": "Original system instruction.",
                "messages": [
                    {"role": "user", "content": "alpha " * 40},
                    {"role": "assistant", "content": "beta " * 40},
                    {"role": "user", "content": "recent"},
                ],
            }
            plan, _ = manual.old_context_summary_plan(body, exact_cache_enabled=False)

            summarized = manual.apply_old_context_summary(body, plan, "compact summary")

            self.assertEqual(summarized["system"][0], {"type": "text", "text": "Original system instruction."})
            self.assertIn("compact summary", summarized["system"][1]["text"])
            self.assertEqual(summarized["messages"], [{"role": "user", "content": "recent"}])

    def test_old_context_summarization_canary_applies_and_holds_out_without_raw_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            secret = "secret old context must not be stored in metadata"
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-old-summary
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
  canary:
    enabled: true
    fraction: 1.0
    salt: test-salt
    unit: source_hash
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": (secret + " ") * 20},
                    {"role": "assistant", "content": "old answer " * 20},
                    {"role": "user", "content": "recent"},
                ],
            }
            cache = {}

            async def fetch(_request):
                return {
                    "summary": "compact durable facts",
                    "summary_input_tokens": 20,
                    "summary_output_tokens": 5,
                    "summary_cost_est_usd": 0.0002,
                    "summary_status_code": 200,
                }

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=cache.get,
                set_cached_summary=lambda key, value: cache.__setitem__(key, value),
                fetch_summary=fetch,
            ))

            self.assertEqual(meta["status"], "applied")
            self.assertEqual(meta["canary"]["cohort"], "canary_applied")
            self.assertTrue(meta["canary"]["selected"])
            self.assertIn("compact durable facts", summarized["system"][0]["text"])
            metadata_json = json.dumps(meta)
            self.assertNotIn(secret, metadata_json)
            self.assertNotIn("compact durable facts", metadata_json)
            self.assertFalse(meta["canary"]["raw_context_included"])
            self.assertFalse(meta["canary"]["raw_summary_included"])
            self.assertIn("source_hash", meta)
            self.assertLessEqual(len(meta["source_hash"]), 12)

            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-old-summary
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
  canary:
    enabled: true
    fraction: 0.0
    salt: test-salt
    unit: source_hash
""",
                encoding="utf-8",
            )
            manual = importlib.reload(crunch_module)

            held_out, holdout_meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=cache.get,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fetch,
            ))

            self.assertIs(held_out, body)
            self.assertEqual(holdout_meta["status"], "skipped")
            self.assertEqual(holdout_meta["reason"], "canary_holdout")
            self.assertEqual(holdout_meta["canary"]["cohort"], "canary_holdout")

    def test_old_context_summarization_tool_aware_canary_preserves_protocol_and_recent_turns(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-tool-aware-summary
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 3
  keep_recent_turns: 2
  max_summary_chars: 1000
  max_source_chars: 20000
  excluded_categories: []
  block_tool_protocol: false
  canary:
    enabled: true
    fraction: 1.0
    holdout_fraction: 0.0
    salt: tool-aware-salt
    unit: source_hash
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                "messages": [
                    {"role": "user", "content": "old planning detail " * 30},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "old-tool", "name": "read_file", "input": {"path": "a.py"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old-tool", "content": "old file contents"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "old assistant visible notes " * 30}]},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "recent-tool", "name": "read_file", "input": {"path": "b.py"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "recent-tool", "content": "recent file contents"}]},
                ],
            }
            original_tool_blocks = [
                json.dumps({"role": msg["role"], "block": block}, sort_keys=True)
                for msg in body["messages"]
                for block in (msg["content"] if isinstance(msg.get("content"), list) else [])
                if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}
            ]
            original_recent_turns = [json.dumps(msg, sort_keys=True) for msg in body["messages"][-2:]]

            async def fetch(_request):
                return {
                    "summary": "compact tool-aware summary",
                    "summary_input_tokens": 20,
                    "summary_output_tokens": 5,
                    "summary_cost_est_usd": 0.0002,
                    "summary_status_code": 200,
                }

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fetch,
            ))

            summarized_tool_blocks = [
                json.dumps({"role": msg["role"], "block": block}, sort_keys=True)
                for msg in summarized["messages"]
                for block in (msg["content"] if isinstance(msg.get("content"), list) else [])
                if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}
            ]
            summarized_recent_turns = [json.dumps(msg, sort_keys=True) for msg in summarized["messages"][-2:]]
            self.assertEqual(meta["status"], "applied")
            self.assertEqual(meta["canary"]["cohort"], "canary_applied")
            self.assertEqual(meta["category"], "tool-result")
            self.assertEqual(original_tool_blocks, summarized_tool_blocks)
            self.assertEqual(original_recent_turns, summarized_recent_turns)
            self.assertTrue(meta["preservation_check"]["ok"])
            self.assertTrue(meta["tool_protocol_blocks_preserved"])
            self.assertTrue(meta["recent_turns_preserved"])
            self.assertIn("compact tool-aware summary", summarized["system"][0]["text"])

    def test_old_context_summarization_tool_aware_holdout_forwards_unchanged(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-tool-aware-summary
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 3
  keep_recent_turns: 2
  excluded_categories: []
  block_tool_protocol: false
  canary:
    enabled: true
    fraction: 0.0
    holdout_fraction: 1.0
    salt: tool-aware-salt
    unit: source_hash
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": "old planning detail " * 30},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "old-tool", "name": "read_file", "input": {"path": "a.py"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old-tool", "content": "old file contents"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "old assistant visible notes " * 30}]},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "recent-tool", "name": "read_file", "input": {"path": "b.py"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "recent-tool", "content": "recent file contents"}]},
                ],
            }

            async def fail_fetch(_request):
                raise AssertionError("holdout must not fetch summary")

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fail_fetch,
            ))

            self.assertIs(summarized, body)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "canary_holdout")
            self.assertEqual(meta["canary"]["cohort"], "canary_holdout")
            self.assertFalse(meta["changed"])
            self.assertFalse(meta["applied"])

    def test_old_context_summarization_thinking_history_skips_even_when_tool_aware(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  keep_recent_turns: 1
  excluded_categories: []
  block_tool_protocol: false
  block_thinking: true
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "private chain"}, {"type": "text", "text": "old answer"}]},
                    {"role": "user", "content": "old text " * 30},
                    {"role": "user", "content": "recent"},
                ]
            }

            plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=False)

            self.assertIsNone(plan)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "thinking-context-blocked")

    def test_old_context_summarization_summary_fetch_failure_forwards_unchanged(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            secret = "raw old context should stay out of metadata"
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-summary-failure
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  canary:
    enabled: true
    fraction: 1.0
    salt: failure-salt
    unit: source_hash
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": (secret + " ") * 30},
                    {"role": "assistant", "content": "old answer " * 30},
                    {"role": "user", "content": "recent"},
                ],
            }

            async def fetch(_request):
                raise RuntimeError("provider unavailable")

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fetch,
            ))

            rendered = json.dumps(meta, sort_keys=True)
            self.assertIs(summarized, body)
            self.assertEqual(meta["status"], "error")
            self.assertEqual(meta["reason"], "summary-fetch-error")
            self.assertEqual(meta["summary_error_type"], "RuntimeError")
            self.assertFalse(meta["changed"])
            self.assertFalse(meta["applied"])
            self.assertNotIn(secret, rendered)

    def test_old_context_summarization_blocks_tool_protocol_context_by_default(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 4
  keep_recent_turns: 1
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file"}]},
                    {"role": "user", "content": "old plain text " * 20},
                    {"role": "user", "content": "recent"},
                ]
            }

            plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=False)

            self.assertIsNone(plan)
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["reason"], "tool-protocol-context-blocked")

    def test_old_context_summarization_safety_stop_bypasses_application(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
old_context_summarization:
  enabled: true
  rule_id: test-old-summary-stop
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
  canary:
    enabled: true
    fraction: 1.0
    salt: stop-salt
    unit: source_hash
  safety_stop:
    enabled: true
    min_outcome_samples: 2
    window: 10
    max_error_rate: 0.5
    max_retry_rate: 1.0
    max_negative_net_savings_rate: 1.0
    max_summary_failure_rate: 1.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            store = Store(str(tmp_path / "calls.sqlite3"))
            for idx in range(2):
                store.log_call(
                    id=f"call-{idx}",
                    created_at=f"2026-06-09T00:00:0{idx}+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=500,
                    latency_ms=10,
                    crunch_json=stable_json({
                        "old_context_summarization": {
                            "rule_id": "test-old-summary-stop",
                            "status": "applied",
                            "estimated_net_savings_usd": 0.001,
                            "canary": {
                                "enabled": True,
                                "cohort": "canary_applied",
                            },
                        }
                    }),
                    retry_count=0,
                )

            async def fail_fetch(_request):
                raise AssertionError("safety stop should bypass before summary fetch")

            body = {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": "old text " * 30},
                    {"role": "assistant", "content": "old answer " * 30},
                    {"role": "user", "content": "recent"},
                ],
            }
            unchanged, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fail_fetch,
                store_obj=store,
            ))

            self.assertIs(unchanged, body)
            self.assertEqual(meta["status"], "bypass")
            self.assertEqual(meta["reason"], manual.LOCAL_CANARY_SAFETY_STOP_REASON)
            self.assertEqual(meta["safety_stop_state"], "stopped")
            self.assertGreaterEqual(meta["safety_stop"]["error_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
