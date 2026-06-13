import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

import agentflow_proxy.router as router_module
from agentflow_proxy.admin import reload_policy_modules
from agentflow_proxy import stats
from agentflow_proxy import codex_app_policy as codex_app_policy_module
from agentflow_proxy.policy_files import policy_file_snapshot, policy_file_status, stage_policy_draft, utc_now
from agentflow_proxy.policy_workbench import apply_validated_policy_draft, rollback_policy_apply
from agentflow_proxy.policy_bundle import (
    MANAGED_POLICY_HMAC_SECRET_ENV,
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    MANAGED_POLICY_VERIFICATION_SECRETS_ENV,
    apply_policy_bundle,
    attach_policy_bundle_provenance,
    build_policy_bundle,
    compare_policy_bundles,
    review_policy_bundle,
    validate_policy_bundle,
)


WORKBENCH_SENSITIVE_FIXTURES = (
    "raw prompt fixture must not leak",
    "raw response fixture must not leak",
    "raw provider body fixture must not leak",
    "raw tool payload fixture must not leak",
    "raw file content fixture must not leak",
    "request-id-fixture-must-not-leak",
    "session-id-fixture-must-not-leak",
    "cache-key-fixture-must-not-leak",
    "api-key-fixture-must-not-leak",
)


def _assert_workbench_payload_is_metadata_only(testcase, payload):
    rendered = json.dumps(payload, sort_keys=True)
    for value in WORKBENCH_SENSITIVE_FIXTURES:
        testcase.assertNotIn(value, rendered)


def _provenance_env(secret: str | None = None) -> dict[str, str]:
    return {
        MANAGED_POLICY_VERIFICATION_SECRET_ENV: secret or "",
        MANAGED_POLICY_VERIFICATION_SECRETS_ENV: "",
        MANAGED_POLICY_HMAC_SECRET_ENV: "",
    }


class PolicyFileStatusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        self.old_home = os.environ.get("HOME")
        os.environ["HOME"] = self.tmp.name
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")
        asyncio.run(reload_policy_modules())

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        if self.old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self.old_home
        asyncio.run(reload_policy_modules())
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_policy_bundle_exports_effective_default_policy_state(self):
        bundle = asyncio.run(build_policy_bundle())

        self.assertEqual(bundle["schema"], "agentflow.policy_bundle.v1")
        self.assertEqual(bundle["generator"]["name"], "agentflow-proxy")
        self.assertEqual(bundle["generator"]["mode"], "local-offline")
        self.assertFalse(bundle["managed_optimizer"]["enabled"])
        self.assertEqual(bundle["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertEqual(bundle["policies"]["summary"]["policy_count"], 5)
        self.assertFalse(bundle["policies"]["summary"]["reload_required"])
        self.assertEqual(bundle["policies"]["summary"]["reload_required_sections"], [])
        self.assertIn("routing", bundle["policies"])
        self.assertIn("crunch", bundle["policies"])
        self.assertIn("cache", bundle["policies"])
        self.assertIn("routing_experiments", bundle["policies"])
        self.assertIn("codex_app", bundle["policies"])
        self.assertIn("openai", bundle["policies"]["routing"])
        self.assertIn("canary", bundle["policies"]["routing"]["openai"])
        self.assertFalse(bundle["policies"]["routing"]["openai"]["canary"]["enabled"])
        self.assertEqual(bundle["policies"]["routing"]["openai"]["rule_path"], bundle["policies"]["routing"]["rule_path"])
        self.assertEqual(bundle["policies"]["crunch"]["pattern_rules"], [])
        self.assertFalse(bundle["policies"]["codex_app"]["review_only"])
        self.assertEqual(bundle["policies"]["codex_app"]["policy_source"], "local-default")
        self.assertIn("file", bundle["policies"]["codex_app"])
        instruction_dedup = bundle["policies"]["crunch"]["instruction_section_deduplication"]
        self.assertFalse(instruction_dedup["enabled"])
        self.assertEqual(instruction_dedup["policy_source"], "local-default")
        self.assertEqual(instruction_dedup["canary"]["fraction"], 0.0)
        self.assertEqual(instruction_dedup["canary"]["holdout_fraction"], 1.0)
        self.assertEqual(instruction_dedup["rules"], [])

    def test_policy_bundle_validation_accepts_exported_bundle(self):
        bundle = asyncio.run(build_policy_bundle())

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["provenance"]["status"], "not-configured")

    def test_policy_bundle_validation_accepts_crunch_pattern_rules(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["crunch"]["pattern_rules"] = [
            {
                "id": "reviewed-scaffold",
                "enabled": True,
                "policy_source": "managed-recommended",
                "candidate_id": "candidate-123",
                "conditions": {
                    "pattern_hashes": [
                        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
                    ],
                    "category": "chat",
                    "min_repeated_count": 2,
                    "keep_recent_matches": 1,
                    "min_text_chars": 1000,
                    "max_applications": 4,
                },
                "action": {
                    "type": "shorten",
                    "head_chars": 80,
                    "tail_chars": 70,
                    "max_replacement_chars": 260,
                },
            }
        ]

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_accepts_repeated_provider_scaffolding_rules(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["crunch"]["repeated_provider_scaffolding"] = {
            "enabled": True,
            "min_request_chars": 12000,
            "min_section_chars": 700,
            "keep_recent_messages": 2,
            "keep_recent_matches": 1,
            "max_replacements": 8,
            "block_tool_protocol": True,
            "block_thinking": True,
            "rules": [
                {
                    "id": "reviewed-provider-scaffold",
                    "enabled": True,
                    "policy_source": "managed-recommended",
                    "candidate_id": "candidate-provider-scaffold",
                    "pattern_hashes": [
                        "sha256:2222222222222222222222222222222222222222222222222222222222222222"
                    ],
                    "min_repeated_count": 2,
                    "keep_recent_matches": 1,
                    "max_applications": 4,
                    "action": {
                        "type": "omit",
                        "max_replacement_chars": 360,
                    },
                }
            ],
        }

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_accepts_instruction_section_dedup_rules(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["crunch"]["instruction_section_deduplication"] = {
            "enabled": True,
            "policy_source": "local-manual",
            "source_surfaces": ["anthropic_messages", "openai_responses", "codex_turn"],
            "categories": ["chat", "long-context"],
            "workflow_phases": ["planning", "verification"],
            "min_section_chars": 900,
            "min_repeated_count": 3,
            "keep_recent_sections": 1,
            "replacement_notice": "[repeated instruction section omitted by AgentFlow]",
            "max_replacements": 4,
            "block_tool_protocol": True,
            "block_tool_payloads": True,
            "block_responses": True,
            "block_thinking": True,
            "canary": {
                "enabled": True,
                "canary_fraction": 0.1,
                "holdout_fraction": 0.9,
                "canary_salt": "local-test",
                "canary_unit": "instruction_section_fingerprint",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 5,
                "window": 500,
                "max_error_rate": 0.1,
                "max_retry_rate": 0.25,
                "max_negative_savings_rate": 0.25,
                "max_error_rate_delta": 0.05,
            },
            "rules": [
                {
                    "id": "synthetic-instruction-dedup",
                    "enabled": True,
                    "policy_source": "local-manual",
                    "instruction_section_fingerprints": [
                        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    ],
                    "source_surfaces": ["anthropic_messages"],
                    "categories": ["chat"],
                    "workflow_phases": ["planning"],
                    "min_section_chars": 900,
                    "min_repeated_count": 3,
                    "keep_recent_sections": 1,
                    "replacement_notice": "[repeated instruction section omitted by AgentFlow]",
                    "max_replacements": 2,
                    "block_tool_protocol": True,
                    "block_tool_payloads": True,
                    "block_responses": True,
                    "block_thinking": True,
                    "action": {"type": "omit_instruction_section"},
                    "canary": {"enabled": True, "canary_fraction": 0.1, "holdout_fraction": 0.9},
                }
            ],
        }

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_manual_crunch_rules_export_instruction_section_dedup_rule(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                """
enabled: true
instruction_section_deduplication:
  enabled: true
  source_surfaces:
    - anthropic_messages
  categories:
    - chat
  workflow_phases:
    - planning
  min_section_chars: 900
  min_repeated_count: 3
  keep_recent_sections: 1
  replacement_notice: "[repeated instruction section omitted by AgentFlow]"
  max_replacements: 2
  canary:
    enabled: true
    canary_fraction: 0.10
    holdout_fraction: 0.90
    canary_salt: synthetic
  safety_stop:
    enabled: true
    min_outcome_samples: 5
    window: 500
  rules:
    - id: synthetic-instruction-dedup
      enabled: true
      instruction_section_fingerprints:
        - sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
      conditions:
        source_surface: anthropic_messages
        category: chat
        workflow_phase: planning
        min_section_chars: 900
        min_repeated_count: 3
      action:
        type: omit_instruction_section
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"AGENTFLOW_CRUNCH_RULES": str(rules_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    bundle = asyncio.run(build_policy_bundle())
                finally:
                    asyncio.run(reload_policy_modules())

        instruction_dedup = bundle["policies"]["crunch"]["instruction_section_deduplication"]
        self.assertTrue(instruction_dedup["enabled"])
        self.assertEqual(instruction_dedup["policy_source"], "local-manual")
        self.assertEqual(instruction_dedup["rules"][0]["id"], "synthetic-instruction-dedup")
        self.assertEqual(
            instruction_dedup["rules"][0]["instruction_section_fingerprints"],
            ["sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
        )
        self.assertEqual(instruction_dedup["rules"][0]["source_surfaces"], ["anthropic_messages"])
        self.assertEqual(instruction_dedup["rules"][0]["canary"]["fraction"], 0.1)
        result = validate_policy_bundle(bundle)
        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_rejects_unsafe_instruction_section_dedup_rules(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["crunch"]["instruction_section_deduplication"] = {
            "enabled": True,
            "policy_source": "managed-enforced",
            "canary": {"canary_fraction": 0.8, "holdout_fraction": 0.4},
            "rules": [
                {
                    "id": "unsafe-instruction-dedup",
                    "enabled": True,
                    "policy_source": "managed-enforced",
                    "instruction_section_fingerprints": ["not-a-sha"],
                    "raw_instruction_text": "raw prompt fixture must not leak",
                    "tenant_id": "tenant-fixture-secret",
                    "request_id": "request-fixture-secret",
                    "cache_key": "cache-fixture-secret",
                    "policy_yaml": "instruction_section_deduplication: raw fixture secret",
                    "block_tool_payloads": False,
                    "action": {"type": "omit_instruction_section", "target": "tool_payload", "provider_body": "raw provider body fixture"},
                    "canary": {"canary_fraction": 1.2},
                }
            ],
        }

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        paths = {error["path"] for error in result["errors"]}
        self.assertIn("$.policies.crunch.instruction_section_deduplication.policy_source", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.canary", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].policy_source", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].instruction_section_fingerprints[0]", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].raw_instruction_text", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].tenant_id", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].request_id", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].cache_key", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].policy_yaml", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].action.provider_body", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].block_tool_payloads", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].action", paths)
        self.assertIn("$.policies.crunch.instruction_section_deduplication.rules[0].canary.canary_fraction", paths)

    def _managed_policy_bundle(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "policy_source": "managed-recommended",
            "candidate_ids": ["candidate-route-chat"],
            "candidate_count": 1,
        }
        bundle["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "note": "Review-only managed recommendation.",
        }
        bundle["policies"]["routing"]["policy_source"] = "managed-recommended"
        bundle["policies"]["routing"].setdefault("rules", []).append({
            "conditions": {
                "model_pattern": "sonnet",
                "category": "chat",
                "has_tools": False,
            },
            "action": {
                "route_to": "claude-haiku-4-5-20251001",
                "reason": "managed candidate for local review",
            },
            "managed_recommendation": {
                "policy_source": "managed-recommended",
                "candidate_id": "candidate-route-chat",
                "confidence": 0.82,
                "sample_count": 24,
            },
        })
        return bundle

    def test_managed_policy_bundle_signed_provenance_verifies_and_reviews(self):
        secret = "test-managed-policy-secret"
        current = asyncio.run(build_policy_bundle())
        signed = attach_policy_bundle_provenance(
            self._managed_policy_bundle(),
            secret=secret,
            issuer="agentflow-server",
            server_id="managed-dev",
            key_id="test-key",
            generated_at="2026-06-08T12:00:00+00:00",
        )

        with patch.dict(os.environ, _provenance_env(secret), clear=False):
            validation = validate_policy_bundle(signed)
            review = review_policy_bundle(current, signed, include_impact=False)

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["provenance"]["status"], "verified")
        self.assertTrue(validation["provenance"]["managed_bundle"])
        self.assertEqual(validation["provenance"]["issuer"], "agentflow-server")
        self.assertTrue(review["ok"])
        self.assertEqual(review["provenance"]["status"], "verified")

    def test_managed_policy_bundle_tampered_after_signing_is_rejected_before_apply(self):
        secret = "test-managed-policy-secret"
        signed = attach_policy_bundle_provenance(
            self._managed_policy_bundle(),
            secret=secret,
            issuer="agentflow-server",
            server_id="managed-dev",
            key_id="test-key",
            generated_at="2026-06-08T12:00:00+00:00",
        )
        signed["policies"]["routing"]["rules"][-1]["action"]["route_to"] = "claude-opus-4-5"

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, _provenance_env(secret), clear=False):
                result = apply_policy_bundle(signed, config_dir=tmp, dry_run=False, allow_risky=True)

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "validation_failed")
            self.assertEqual(result["provenance"]["status"], "invalid")
            self.assertIn("$.provenance.bundle_hash", {error["path"] for error in result["validation"]["errors"]})
            self.assertFalse(list(Path(tmp).glob("*.yaml")))

    def test_managed_policy_bundle_missing_provenance_is_rejected_when_verification_configured(self):
        with patch.dict(os.environ, _provenance_env("test-managed-policy-secret"), clear=False):
            result = validate_policy_bundle(self._managed_policy_bundle())

        self.assertFalse(result["ok"])
        self.assertEqual(result["provenance"]["status"], "missing")
        self.assertIn("$.provenance", {error["path"] for error in result["errors"]})

    def test_managed_policy_bundle_reports_not_configured_without_secret(self):
        with patch.dict(os.environ, _provenance_env(), clear=False):
            result = validate_policy_bundle(self._managed_policy_bundle())

        self.assertTrue(result["ok"])
        self.assertEqual(result["provenance"]["status"], "not-configured")
        self.assertTrue(result["provenance"]["managed_bundle"])
        self.assertTrue(result["warnings"])

    def test_local_unsigned_policy_bundle_remains_valid_with_verification_secret(self):
        bundle = asyncio.run(build_policy_bundle())

        with patch.dict(os.environ, _provenance_env("test-managed-policy-secret"), clear=False):
            result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["provenance"]["status"], "missing")
        self.assertFalse(result["provenance"]["managed_bundle"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_rejects_missing_policy_section(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"].pop("cache")

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        self.assertIn("$.policies.cache", {error["path"] for error in result["errors"]})

    def test_policy_bundle_validation_rejects_malformed_section_shapes(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["routing"]["rules"][0]["conditions"]["text_chars_lt"] = "many"
        bundle["policies"]["routing"]["rules"][0]["conditions"]["unsupported"] = True
        bundle["policies"]["routing"]["rules"][0]["action"]["route_to"] = ""
        bundle["policies"]["crunch"]["old_context_summarization"]["placement"] = "message"
        bundle["policies"]["crunch"]["thinking_deduplication"]["similarity_threshold"] = 1.5
        bundle["policies"]["cache"]["semantic_cache"]["threshold"] = -0.1
        bundle["policies"]["cache"]["file_watch"]["max_paths"] = "lots"

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        paths = {error["path"] for error in result["errors"]}
        self.assertIn("$.policies.routing.rules[0].conditions.text_chars_lt", paths)
        self.assertIn("$.policies.routing.rules[0].conditions.unsupported", paths)
        self.assertIn("$.policies.routing.rules[0].action.route_to", paths)
        self.assertIn("$.policies.crunch.old_context_summarization.placement", paths)
        self.assertIn("$.policies.crunch.thinking_deduplication.similarity_threshold", paths)
        self.assertIn("$.policies.cache.semantic_cache.threshold", paths)
        self.assertIn("$.policies.cache.file_watch.max_paths", paths)

    def test_policy_bundle_validation_rejects_unsafe_openai_canary_policy(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["routing"]["openai"]["canary"] = {
            "enabled": True,
            "policy_id": "unsafe-openai-canary",
            "model_pattern": "gpt-5",
            "target_model": "claude-haiku-4-5-20251001",
            "eligible_categories": ["unknown-category"],
            "excluded_categories": ["chat"],
            "canary_fraction": 0.8,
            "holdout_fraction": 0.4,
            "raw_prompt": "must never be accepted",
        }

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        paths = {error["path"] for error in result["errors"]}
        self.assertIn("$.policies.routing.openai.canary.target_model", paths)
        self.assertIn("$.policies.routing.openai.canary.eligible_categories[0]", paths)
        self.assertIn("$.policies.routing.openai.canary.raw_prompt", paths)
        self.assertIn("$.policies.routing.openai.canary", paths)

    def test_apply_policy_bundle_writes_openai_canary_to_routing_yaml(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["routing"]["openai"]["canary"] = {
            "enabled": True,
            "policy_id": "reviewed-openai-canary",
            "model_pattern": "gpt-5",
            "target_model": "gpt-5.4-mini",
            "eligible_categories": ["chat"],
            "excluded_categories": [],
            "allow_tools": False,
            "allow_stream": False,
            "min_text_chars": 0,
            "max_text_chars": 8000,
            "min_input_tokens_est": 0,
            "max_input_tokens_est": 2000,
            "canary_fraction": 0.1,
            "holdout_fraction": 0.1,
            "salt": "reviewed-openai-canary-salt",
            "safety_stop": {"enabled": False},
        }

        with TemporaryDirectory() as tmp:
            result = apply_policy_bundle(bundle, config_dir=tmp, dry_run=False)
            data = yaml.safe_load((Path(tmp) / "routing_rules.yaml").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(data["openai_canary"]["policy_id"], "reviewed-openai-canary")
        self.assertEqual(data["openai_canary"]["target_model"], "gpt-5.4-mini")
        self.assertEqual(data["openai_canary"]["canary_fraction"], 0.1)
        self.assertEqual(data["openai_canary"]["holdout_fraction"], 0.1)

    def test_policy_bundle_validation_accepts_reviewable_codex_app_policy(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"] = {
            "enabled": True,
            "policy_source": "managed-recommended",
            "surface": "codex_app_turn",
            "review_only": True,
            "rules": [
                {
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "missing",
                        "input_size_bucket": "medium",
                        "cache_eligible": False,
                        "replayability_level": "turn-metadata-only",
                        "has_action_like_params": False,
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "crunch_profile": "codex-summary",
                        "cache_eligible": False,
                        "cache_eligibility_reason": "metadata-only recommendation requires local replayability review",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                }
            ],
        }

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_rejects_codex_app_managed_enforced(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"]["policy_source"] = "managed-enforced"

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        self.assertIn("$.policies.codex_app.policy_source", {error["path"] for error in result["errors"]})

    def test_policy_review_reports_codex_app_section_without_provider_routing_apply(self):
        current = asyncio.run(build_policy_bundle())
        proposed = json.loads(json.dumps(current))
        proposed["policies"]["codex_app"]["policy_source"] = "managed-recommended"
        proposed["policies"]["codex_app"]["rules"] = [
            {
                "candidate_id": "codex-tool-execution-small",
                "conditions": {
                    "app_family": "codex",
                    "workflow_phase": "tool_execution",
                    "model_field_state": "present",
                    "input_size_bucket": "small",
                    "cache_eligible": True,
                    "replayability_level": "local-exact-response",
                },
                "action": {
                    "recommended_model": "gpt-5-mini",
                    "crunch_profile": "pass-through",
                    "cache_eligibility_reason": "local exact replay only",
                    "reason": "managed Codex app turn policy for local review",
                },
            }
        ]

        result = review_policy_bundle(current, proposed, include_impact=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["changed_sections"], ["codex_app"])
        self.assertTrue(any(change["path"].startswith("$.policies.codex_app") for change in result["diff"]["changes"]))
        codex_review = result["section_reviews"]["codex_app"]
        self.assertEqual(codex_review["status"], "review-only")
        self.assertTrue(codex_review["review_only"])
        self.assertEqual(codex_review["candidate_ids"], ["codex-tool-execution-small"])
        self.assertEqual(codex_review["application"]["status"], "not-applied")
        self.assertFalse(codex_review["application"]["writes_local_policy_files"])
        self.assertIn("workflow_phase", codex_review["condition_keys_present"])
        self.assertIn("recommended_model", codex_review["action_keys_present"])
        self.assertFalse(codex_review["privacy"]["raw_prompts_included"])

    def test_policy_review_reports_managed_pattern_candidate_evidence_without_raw_leakage(self):
        current = asyncio.run(build_policy_bundle())
        proposed = json.loads(json.dumps(current))
        proposed["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "policy_source": "managed-recommended",
            "candidate_ids": [
                "pattern-crunch-representable",
                "pattern-cache-health-changed",
                "pattern-cache-omitted",
                "pattern-cache-unchanged",
            ],
            "candidate_count": 4,
            "change_summary": {
                "since_bundle_hash": "sha256:last-reviewed",
                "changed_candidate_ids": ["pattern-cache-health-changed"],
                "unchanged_candidate_ids": ["pattern-cache-unchanged"],
            },
        }
        proposed["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "note": "Review-only managed pattern evidence.",
        }
        proposed["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-crunch-representable",
                    "candidate_family": "crunch-policy-rule",
                    "confidence": 0.78,
                    "sample_count": 42,
                    "estimated_savings_usd": 1.25,
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "text_bucket": "8k_30k_chars",
                    "token_bucket": "1k_4k_tokens",
                    "action": {
                        "crunch_profile": "repeated-section-dedupe",
                        "command": "raw shell command must not render",
                    },
                    "local_action_requirements": {
                        "expected_policy_section": "crunch",
                        "actionability_status": "review-only-local-action",
                        "action_requirements": {"crunch_profile_support": True},
                    },
                    "confidence_inputs": {
                        "score_family": "crunch-policy-rule",
                        "privacy_profile_counts": {"metadata-only": 42},
                    },
                    "review_evidence": {
                        "crunch": {"saved_tokens_est": 3200},
                        "raw_prompt": "raw prompt must not render",
                    },
                }
            ],
        }
        proposed["policies"]["cache"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 2,
            "omitted_candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-cache-health-changed",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.65,
                    "sample_count": 18,
                    "estimated_savings_usd": 0.12,
                    "delta": {"status": "changed-health", "old_health": "warning", "new_health": "healthy"},
                    "warning_reasons": ["lifecycle-warning-cleared"],
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {
                        "review_evidence": {"cache": {"blocker": "tool_calls_without_file_dependencies"}},
                        "api_key": "secret must not render",
                    },
                },
                {
                    "candidate_id": "pattern-cache-unchanged",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.51,
                    "sample_count": 11,
                    "delta": {"status": "unchanged"},
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                },
            ],
            "omitted_candidates": [
                {
                    "candidate_id": "pattern-cache-omitted",
                    "candidate_family": "cache-policy-rule",
                    "reason": "cache-policy-rule-not-representable-in-local-bundle-schema-yet",
                    "omission_reasons": ["cache-policy-rule-not-representable-in-local-bundle-schema-yet"],
                    "sample_count": 9,
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {"raw_response": "raw provider response must not render"},
                }
            ],
        }

        result = review_policy_bundle(current, proposed, include_impact=False)
        rendered = json.dumps(result, sort_keys=True)

        self.assertTrue(result["ok"])
        self.assertIn("crunch", result["section_reviews"])
        self.assertIn("cache", result["section_reviews"])
        crunch_review = result["section_reviews"]["crunch"]
        cache_review = result["section_reviews"]["cache"]
        self.assertEqual(crunch_review["schema"], "agentflow.pattern_candidate_review.v1")
        self.assertEqual(crunch_review["candidate_count"], 1)
        self.assertEqual(crunch_review["representable_candidate_count"], 1)
        self.assertEqual(crunch_review["candidates"][0]["sample_count_bucket"], "25_99")
        self.assertEqual(crunch_review["candidates"][0]["savings_bucket"], "gte_1_usd")
        self.assertEqual(cache_review["candidate_count"], 3)
        self.assertEqual(cache_review["changed_health_candidate_count"], 1)
        self.assertEqual(cache_review["unchanged_candidate_count"], 1)
        self.assertEqual(cache_review["omitted_candidate_count"], 1)
        self.assertEqual(cache_review["application"]["status"], "review-only-not-applied")
        self.assertIn("cache pattern candidates: 3 total", " ".join(result["human_summary"]))
        self.assertNotIn("raw prompt must not render", rendered)
        self.assertNotIn("raw shell command must not render", rendered)
        self.assertNotIn("secret must not render", rendered)
        self.assertNotIn("raw provider response must not render", rendered)
        self.assertNotIn("api_key", rendered)

    def test_policy_apply_dry_run_reports_reviewed_codex_app_yaml_without_writing(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"] = {
            **bundle["policies"]["codex_app"],
            "policy_source": "managed-recommended",
            "rules": [
                {
                    "candidate_id": "codex-summary-pass-through",
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            result = apply_policy_bundle(bundle, config_dir=tmp, dry_run=True)

            self.assertTrue(result["ok"])
            self.assertEqual(result["codex_app"]["status"], "dry-run")
            self.assertEqual(result["codex_app"]["selected_candidate_ids"], ["codex-summary-pass-through"])
            self.assertFalse((Path(tmp) / "codex_app_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())
            codex_file = next(file for file in result["files"] if file["section"] == "codex_app")
            self.assertTrue(codex_file["changed"])
            self.assertIn("codex-summary-pass-through", codex_file["diff"])
            self.assertIn("policy_source: managed-recommended", codex_file["diff"])

    def test_codex_app_manual_policy_file_exports_and_applies_safe_actions(self):
        try:
            with TemporaryDirectory() as tmp:
                rules_path = Path(tmp) / "codex_app_rules.yaml"
                rules_path.write_text(
                    """
enabled: true
summary_model_hint:
  enabled: true
  target_model: gpt-5-mini
exact_cache:
  enabled: true
  namespace: reviewable-codex
crunch:
  profiles:
    - codex-repeated-scaffolding
""",
                    encoding="utf-8",
                )
                with patch.dict(
                    os.environ,
                    {
                        "AGENTFLOW_CODEX_APP_RULES": str(rules_path),
                        "HOME": tmp,
                        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT": "0",
                        "AGENTFLOW_CODEX_APP_CACHE": "0",
                    },
                    clear=False,
                ):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(stats)
                    bundle = asyncio.run(build_policy_bundle())

                codex_policy = bundle["policies"]["codex_app"]
                surface = bundle["policies"]["source_surfaces"]["codex_turn"]
                self.assertEqual(codex_policy["policy_source"], "local-manual")
                self.assertEqual(codex_policy["rule_path"], str(rules_path))
                self.assertFalse(codex_policy["file"]["reload_required"])
                self.assertTrue(codex_policy["summary_model_hint"]["enabled"])
                self.assertEqual(codex_policy["summary_model_hint"]["target_model"], "gpt-5-mini")
                self.assertTrue(codex_policy["exact_cache"]["enabled"])
                self.assertEqual(codex_policy["exact_cache"]["namespace"], "reviewable-codex")
                self.assertEqual(surface["routing"]["summary_model_hint"]["policy_source"], "local-manual")
                self.assertEqual(surface["cache"]["policy_source"], "local-manual")
                self.assertEqual(bundle["policies"]["summary"]["manual_policy_count"], 1)

                with TemporaryDirectory() as apply_tmp:
                    result = apply_policy_bundle(bundle, config_dir=apply_tmp, dry_run=True, sections=["codex_app"])

                self.assertTrue(result["ok"])
                self.assertEqual(result["applied_sections"], ["codex_app"])
                self.assertTrue(all(item["reason"] == "not-requested" for item in result["skipped_sections"]))
                codex_file = result["files"][0]
                self.assertEqual(codex_file["section"], "codex_app")
                self.assertEqual(Path(codex_file["path"]).name, "codex_app_rules.yaml")
        finally:
            asyncio.run(reload_policy_modules())

    def test_codex_app_default_terminal_transcript_compaction_is_disabled_and_valid(self):
        bundle = asyncio.run(build_policy_bundle())

        validation = validate_policy_bundle(bundle)
        codex_policy = bundle["policies"]["codex_app"]
        terminal = codex_policy["terminal_transcript_compaction"]

        self.assertTrue(validation["ok"])
        self.assertFalse(terminal["enabled"])
        self.assertTrue(terminal["review_only"])
        self.assertFalse(terminal["runtime_mutation_enabled"])
        self.assertEqual(terminal["application"]["status"], "review-only-not-applied")
        self.assertEqual(terminal["rule_count"], 0)
        self.assertEqual(terminal["conditions"]["source_surface"], "codex_turn")
        self.assertEqual(terminal["conditions"]["workflow_phase"], "tool_execution")
        self.assertEqual(terminal["canary"]["fraction"], 0.0)
        self.assertEqual(terminal["canary"]["holdout_fraction"], 1.0)
        self.assertFalse(terminal["canary"]["salt_configured"])
        self.assertNotIn("salt", terminal["canary"])

    def test_codex_app_terminal_transcript_policy_file_exports_sanitized_public_metadata(self):
        raw_candidate_id = "/workspace/private/raw-codex-terminal-candidate-must-not-leak"
        raw_action_id = "raw-action-id-must-not-leak"
        raw_salt = "secret-terminal-canary-salt-must-not-leak"
        raw_rule_path = "/workspace/private/raw-path-must-not-leak"
        try:
            with TemporaryDirectory() as tmp:
                rules_path = Path(tmp) / "codex_app_rules.yaml"
                rules_path.write_text(
                    f"""
enabled: true
summary_model_hint:
  enabled: false
exact_cache:
  enabled: false
terminal_transcript_compaction:
  enabled: true
  review_only: true
  policy_source: local-manual
  rule_id: codex-terminal-review-1
  candidate_id: "{raw_candidate_id}"
  action_id: "{raw_action_id}"
  conditions:
    source_surface: codex_turn
    app_family: codex
    granularity: agent_turn
    workflow_phase: tool_execution
    text_bucket:
      - 32k_128k_chars
    terminal_fraction_bucket:
      - gte_75pct
    terminal_event_count_bucket:
      - 21_100
    terminal_signal_source: input-terminal-features+event-window
    cache_status: skipped
    already_crunched_repeated_scaffold: false
    safety_preserve_diagnostics: true
    min_input_chars: 8000
    min_terminal_chars: 2000
    min_projected_saved_chars: 750
  action:
    type: compact_terminal_transcript
    keep_recent_turns: 3
    min_block_chars: 2500
    head_lines: 10
    tail_lines: 18
    max_evidence_lines: 90
    min_saved_chars: 750
    preserve_diagnostics: true
    preserve_tool_protocol: true
    preserve_recent_turns: true
    preserve_error_lines: true
  canary:
    enabled: true
    canary_fraction: 0.15
    holdout_fraction: 0.85
    canary_salt: "{raw_salt}"
    canary_unit: source_hash
  safety_stop:
    enabled: true
    min_outcome_samples: 9
    window: 700
    max_error_rate: 0.04
    max_retry_rate: 0.12
    max_negative_savings_rate: 0.18
    max_error_rate_delta: 0.02
  provenance:
    schema: agentflow.codex_terminal_transcript_compaction_policy.v1
    issuer: local-agentflow
    server_id: "{raw_rule_path}"
    decision_hash: "{raw_rule_path}"
    verified: true
  rules:
    - id: codex-terminal-review-rule-1
      enabled: true
      candidate_id: codex-terminal-candidate-safe
      action_id: codex-terminal-action-safe
      conditions:
        source_surface: codex_turn
        workflow_phase: tool_execution
        terminal_fraction_bucket: gte_75pct
      action:
        type: compact_terminal_transcript
        min_saved_chars: 900
      canary:
        enabled: true
        fraction: 0.0
        holdout_fraction: 1.0
        salt: "{raw_salt}"
        unit: source_hash
""",
                    encoding="utf-8",
                )
                with patch.dict(os.environ, {"AGENTFLOW_CODEX_APP_RULES": str(rules_path), "HOME": tmp}, clear=False):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(stats)
                    bundle = asyncio.run(build_policy_bundle())
                    effective = codex_app_policy_module.codex_terminal_transcript_compaction_effective_policy()

                validation = validate_policy_bundle(bundle)
                codex_policy = bundle["policies"]["codex_app"]
                terminal = codex_policy["terminal_transcript_compaction"]
                surface_terminal = bundle["policies"]["source_surfaces"]["codex_turn"]["terminal_transcript_compaction"]

                self.assertTrue(validation["ok"])
                self.assertTrue(terminal["enabled"])
                self.assertTrue(terminal["review_only"])
                self.assertFalse(terminal["runtime_mutation_enabled"])
                self.assertEqual(terminal["policy_source"], "local-manual")
                self.assertEqual(terminal["rule_id"], "codex-terminal-review-1")
                self.assertTrue(str(terminal["candidate_id"]).startswith("codex-terminal-transcript-candidate:"))
                self.assertTrue(str(terminal["action_id"]).startswith("codex-terminal-transcript-action:"))
                self.assertEqual(terminal["conditions"]["workflow_phase"], "tool_execution")
                self.assertEqual(terminal["conditions"]["min_projected_saved_chars"], 750)
                self.assertEqual(terminal["action"]["keep_recent_turns"], 3)
                self.assertEqual(terminal["action"]["min_saved_chars"], 750)
                self.assertEqual(terminal["canary"]["fraction"], 0.15)
                self.assertEqual(terminal["canary"]["holdout_fraction"], 0.85)
                self.assertTrue(terminal["canary"]["salt_configured"])
                self.assertNotIn("salt", terminal["canary"])
                self.assertEqual(terminal["safety_stop"]["min_outcome_samples"], 9)
                self.assertEqual(terminal["rule_count"], 1)
                self.assertEqual(terminal["rules"][0]["candidate_id"], "codex-terminal-candidate-safe")
                self.assertEqual(surface_terminal["candidate_id"], terminal["candidate_id"])
                self.assertEqual(effective["candidate_id"], terminal["candidate_id"])

                rendered = json.dumps(terminal, sort_keys=True)
                self.assertNotIn(raw_candidate_id, rendered)
                self.assertNotIn(raw_action_id, rendered)
                self.assertNotIn(raw_salt, rendered)
                self.assertNotIn(raw_rule_path, rendered)
                self.assertNotIn(str(rules_path), rendered)
                self.assertNotIn("/workspace/private", rendered)

                with TemporaryDirectory() as apply_tmp:
                    result = apply_policy_bundle(bundle, config_dir=apply_tmp, dry_run=True, sections=["codex_app"])

                self.assertTrue(result["ok"])
                codex_file = result["files"][0]
                self.assertIn("terminal_transcript_compaction", codex_file["diff"])
                self.assertIn("review_only: true", codex_file["diff"])
                self.assertNotIn(raw_salt, codex_file["diff"])
        finally:
            asyncio.run(reload_policy_modules())

    def test_codex_terminal_transcript_validation_errors_do_not_echo_raw_policy_keys(self):
        raw_condition_key = "/workspace/private/raw-terminal-condition-key-must-not-leak"
        raw_action_key = "/workspace/private/raw-terminal-action-key-must-not-leak"
        raw_condition_value = "raw terminal condition value must not leak"
        raw_action_value = "raw terminal action value must not leak"
        bundle = asyncio.run(build_policy_bundle())
        terminal = bundle["policies"]["codex_app"]["terminal_transcript_compaction"]
        terminal["conditions"][raw_condition_key] = raw_condition_value
        terminal["action"][raw_action_key] = raw_action_value

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertIn("unknown Codex terminal-transcript condition", rendered)
        self.assertIn("unknown Codex terminal-transcript action", rendered)
        self.assertIn(".conditions.redacted", rendered)
        self.assertIn(".action.redacted", rendered)
        for forbidden in (
            raw_condition_key,
            raw_action_key,
            raw_condition_value,
            raw_action_value,
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_policy_bundle_compare_reports_no_changes_for_identical_bundles(self):
        bundle = asyncio.run(build_policy_bundle())

        result = compare_policy_bundles(bundle, json.loads(json.dumps(bundle)))

        self.assertTrue(result["ok"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["change_count"], 0)
        self.assertEqual(result["changed_sections"], [])

    def test_policy_bundle_compare_reports_policy_section_changes(self):
        before = asyncio.run(build_policy_bundle())
        after = json.loads(json.dumps(before))
        after["policies"]["routing"]["rules"][0]["action"]["reason"] = "changed in diff test"
        after["policies"]["cache"]["semantic_cache"]["threshold"] = 0.99

        result = compare_policy_bundles(before, after)

        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertEqual(result["change_count"], 2)
        self.assertEqual(result["changed_sections"], ["cache", "routing"])
        changes_by_path = {change["path"]: change for change in result["changes"]}
        self.assertEqual(
            changes_by_path["$.policies.routing.rules[0].action.reason"]["new"],
            "changed in diff test",
        )
        self.assertEqual(changes_by_path["$.policies.cache.semantic_cache.threshold"]["new"], 0.99)

    def test_policy_bundle_compare_returns_validation_errors_for_bad_input(self):
        bundle = asyncio.run(build_policy_bundle())

        result = compare_policy_bundles({"schema": "wrong"}, bundle)

        self.assertFalse(result["ok"])
        self.assertFalse(result["changed"])
        self.assertIn("$.schema", {error["path"] for error in result["before_validation"]["errors"]})

    def test_policy_draft_stages_section_diff_without_touching_active_yaml(self):
        with TemporaryDirectory() as tmp:
            active_path = Path(tmp) / "cache_rules.yaml"
            active_text = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            active_path.write_text(active_text, encoding="utf-8")
            workspace = Path(tmp) / "drafts"

            with patch.dict(os.environ, {"AGENTFLOW_CACHE_RULES": str(active_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    result = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"enabled": True, "threshold": 0.91}},
                        section="cache",
                        draft_id="cache-threshold-review",
                        workspace=workspace,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(result["ok"])
            self.assertEqual(result["schema"], "agentflow.policy_draft_stage.v1")
            self.assertFalse(result["wrote_active_policy_files"])
            self.assertFalse(result["reloaded_modules"])
            self.assertFalse(result["provider_calls_made"])
            self.assertEqual(active_path.read_text(encoding="utf-8"), active_text)
            self.assertEqual(result["diff"]["changed_sections"], ["cache"])
            cache_section = {section["section"]: section for section in result["sections"]}["cache"]
            self.assertTrue(cache_section["changed"])
            self.assertEqual(cache_section["target_file"], str(active_path))
            self.assertTrue(cache_section["reload_required_after_apply"])
            self.assertIn("$.policies.cache.semantic_cache.threshold", {change["path"] for change in cache_section["changes"]})
            self.assertTrue((workspace / "cache-threshold-review" / "policy_bundle.json").exists())
            draft_cache_yaml = workspace / "cache-threshold-review" / "sections" / "cache_rules.yaml"
            self.assertTrue(draft_cache_yaml.exists())
            staged = yaml.safe_load(draft_cache_yaml.read_text(encoding="utf-8"))
            self.assertEqual(staged["semantic_cache"]["threshold"], 0.91)

    def test_policy_draft_rejects_raw_prompt_payloads_before_writing_workspace(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"

            result = asyncio.run(stage_policy_draft(
                {"semantic_cache": {"enabled": True}, "raw_request": {"prompt": "do not stage"}},
                section="cache",
                draft_id="unsafe",
                workspace=workspace,
            ))

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "raw_payload_rejected")
            self.assertFalse(result["wrote_active_policy_files"])
            self.assertFalse(workspace.exists())
            self.assertIn("$.raw_request", {error["path"] for error in result["error"]["errors"]})

    def test_policy_draft_rejects_privacy_sensitive_identifiers_payloads_and_secrets(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            payload = {
                "semantic_cache": {"enabled": True},
                "request_id": "request-id-fixture-must-not-leak",
                "session_id": "session-id-fixture-must-not-leak",
                "cache_key": "cache-key-fixture-must-not-leak",
                "api_key": "api-key-fixture-must-not-leak",
                "tool_payload": {"content": "raw tool payload fixture must not leak"},
                "file_contents": "raw file content fixture must not leak",
                "nested": {
                    "raw_prompt": "raw prompt fixture must not leak",
                    "raw_response": "raw response fixture must not leak",
                    "provider_body": "raw provider body fixture must not leak",
                },
            }

            result = asyncio.run(stage_policy_draft(
                payload,
                section="cache",
                draft_id="unsafe-fixtures",
                workspace=workspace,
            ))

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "raw_payload_rejected")
            self.assertFalse(result["wrote_active_policy_files"])
            self.assertFalse(result["provider_calls_made"])
            self.assertFalse(result["managed_server_calls_made"])
            self.assertFalse(workspace.exists())
            paths = {error["path"] for error in result["error"]["errors"]}
            for expected in (
                "$.request_id",
                "$.session_id",
                "$.cache_key",
                "$.api_key",
                "$.tool_payload",
                "$.file_contents",
                "$.nested.raw_prompt",
                "$.nested.raw_response",
                "$.nested.provider_body",
            ):
                self.assertIn(expected, paths)
            _assert_workbench_payload_is_metadata_only(self, result)

    def test_policy_draft_validation_failure_fixture_blocks_apply_without_active_writes(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            draft_dir = workspace / "invalid-cache"
            draft_dir.mkdir(parents=True)
            exported = asyncio.run(build_policy_bundle())
            exported["policies"]["cache"]["semantic_cache"]["threshold"] = "raw prompt fixture must not leak"
            (draft_dir / "policy_bundle.json").write_text(json.dumps(exported), encoding="utf-8")
            (draft_dir / "draft.json").write_text(
                json.dumps({
                    "schema": "agentflow.policy_draft.v1",
                    "draft_id": "invalid-cache",
                    "bundle_path": str(draft_dir / "policy_bundle.json"),
                    "changed": True,
                    "changed_sections": ["cache"],
                    "sections": [{"section": "cache"}],
                }),
                encoding="utf-8",
            )

            validation = asyncio.run(apply_validated_policy_draft(
                "invalid-cache",
                workspace=workspace,
                config_dir=config_dir,
                sections=["cache"],
            ))

            self.assertFalse(validation["ok"])
            self.assertEqual(validation["status"], "blocked")
            self.assertEqual(validation["error"]["type"], "apply_blocked")
            self.assertFalse(validation["reloaded_modules"])
            self.assertFalse(validation["restored"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), original)
            self.assertFalse(list(config_dir.glob("cache_rules.yaml.bak-*")))
            _assert_workbench_payload_is_metadata_only(self, validation)

    def test_policy_draft_apply_transaction_writes_backup_reloads_and_verifies(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"

            env = {
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-apply",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])

                    result = asyncio.run(apply_validated_policy_draft(
                        "cache-apply",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                        event_source="test",
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(result["ok"])
            self.assertEqual(result["schema"], "agentflow.policy_draft_apply.v1")
            self.assertEqual(result["status"], "applied")
            self.assertEqual(result["changed_sections"], ["cache"])
            self.assertTrue(result["reloaded_modules"])
            self.assertTrue(result["verification"]["ok"])
            self.assertFalse(result["restored"])
            self.assertIn("agentflow-policy-rollback", result["rollback_command"])
            rendered = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(rendered["semantic_cache"]["threshold"], 0.91)
            backup_path = Path(result["backups"][0]["path"])
            self.assertTrue(backup_path.exists())
            self.assertEqual(backup_path.read_text(encoding="utf-8"), original)
            events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
            event = [item for item in events if item.get("action") == "draft-apply"][-1]
            self.assertEqual(event["action"], "draft-apply")
            self.assertTrue(event["ok"])
            self.assertEqual(event["details"]["apply_id"], result["apply_id"])
            self.assertFalse(event["details"]["provider_calls_made"])
            self.assertFalse(event["details"]["managed_server_calls_made"])

    def test_policy_draft_apply_restores_file_when_reload_fails(self):
        async def failing_reload():
            raise RuntimeError("reload unavailable")

        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"

            with patch.dict(os.environ, {"AGENTFLOW_CACHE_RULES": str(cache_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-reload-fail",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])

                    result = asyncio.run(apply_validated_policy_draft(
                        "cache-reload-fail",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=failing_reload,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "reload_failed")
            self.assertTrue(result["restored"])
            self.assertTrue(result["restore"]["ok"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), original)

    def test_policy_draft_apply_restores_file_when_verification_fails(self):
        async def stale_reload():
            return {
                "ok": True,
                "reloaded_modules": ["agentflow_proxy.cache"],
                "policies": {
                    "cache": {
                        "file": {
                            "loaded": {"sha256": "wrong"},
                            "current": {"sha256": "wrong"},
                            "reload_required": False,
                        }
                    }
                },
            }

        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"

            with patch.dict(os.environ, {"AGENTFLOW_CACHE_RULES": str(cache_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-verify-fail",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])

                    result = asyncio.run(apply_validated_policy_draft(
                        "cache-verify-fail",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=stale_reload,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "verification_failed")
            self.assertFalse(result["verification"]["ok"])
            self.assertTrue(result["restored"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), original)

    def test_policy_draft_apply_restores_prior_file_when_later_write_fails(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            crunch_path = config_dir / "crunch_rules.yaml"
            cache_path = config_dir / "cache_rules.yaml"
            crunch_original = (
                "enabled: true\n"
                "threshold_chars: 24000\n"
                "prompt_cache:\n"
                "  enabled: true\n"
                "  min_chars: 4096\n"
                "thinking_deduplication:\n"
                "  enabled: true\n"
                "  min_chars: 2000\n"
                "  similarity_threshold: 0.95\n"
            )
            cache_original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            crunch_path.write_text(crunch_original, encoding="utf-8")
            cache_path.write_text(cache_original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"

            env = {
                "AGENTFLOW_CRUNCH_RULES": str(crunch_path),
                "AGENTFLOW_CACHE_RULES": str(cache_path),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    proposed = asyncio.run(build_policy_bundle())
                    proposed["policies"]["crunch"]["threshold_chars"] = 12000
                    proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91
                    stage = asyncio.run(stage_policy_draft(
                        proposed,
                        draft_id="partial-write-failure",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])

                    real_write = __import__("agentflow_proxy.policy_workbench", fromlist=["_atomic_write_policy_text"])._atomic_write_policy_text
                    writes = []
                    failed_cache_write = False

                    def fail_second_write(path, text):
                        nonlocal failed_cache_write
                        writes.append(Path(path).name)
                        if Path(path).name == "cache_rules.yaml" and not failed_cache_write:
                            failed_cache_write = True
                            raise OSError("cache write fixture failure")
                        return real_write(path, text)

                    with patch("agentflow_proxy.policy_workbench._atomic_write_policy_text", side_effect=fail_second_write):
                        result = asyncio.run(apply_validated_policy_draft(
                            "partial-write-failure",
                            workspace=workspace,
                            config_dir=config_dir,
                            sections=["crunch", "cache"],
                            reload_policy_state=reload_policy_modules,
                        ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "write_failed")
            self.assertEqual(writes[:2], ["crunch_rules.yaml", "cache_rules.yaml"])
            self.assertTrue(result["restored"])
            self.assertTrue(result["restore"]["ok"])
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), crunch_original)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), cache_original)
            _assert_workbench_payload_is_metadata_only(self, result)

    def test_policy_draft_rollback_by_apply_id_dry_run_reports_exact_files(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"

            env = {
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-rollback-dry-run",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])
                    apply_result = asyncio.run(apply_validated_policy_draft(
                        "cache-rollback-dry-run",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                    ))
                    self.assertTrue(apply_result["ok"])

                    rollback = asyncio.run(rollback_policy_apply(
                        apply_result["apply_id"],
                        config_dir=config_dir,
                        sections=["cache"],
                        dry_run=True,
                        reload_policy_state=reload_policy_modules,
                        event_source="test",
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(rollback["ok"])
            self.assertEqual(rollback["schema"], "agentflow.policy_draft_rollback.v1")
            self.assertEqual(rollback["status"], "dry-run")
            self.assertEqual(rollback["apply_id"], apply_result["apply_id"])
            self.assertEqual(rollback["restored_sections"], ["cache"])
            self.assertEqual(rollback["files"][0]["restored_from"], apply_result["backups"][0]["path"])
            self.assertTrue(rollback["files"][0]["changed"])
            self.assertFalse(rollback["reloaded_modules"])
            self.assertEqual(yaml.safe_load(cache_path.read_text(encoding="utf-8"))["semantic_cache"]["threshold"], 0.91)

    def test_policy_draft_rollback_by_apply_id_restores_reloads_and_verifies(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"

            env = {
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-rollback",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])
                    apply_result = asyncio.run(apply_validated_policy_draft(
                        "cache-rollback",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                    ))
                    self.assertTrue(apply_result["ok"])
                    self.assertIn("--apply-id", apply_result["rollback_command"])

                    rollback = asyncio.run(rollback_policy_apply(
                        apply_result["apply_id"],
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                        event_source="test",
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(rollback["ok"])
            self.assertEqual(rollback["status"], "rolled-back")
            self.assertEqual(rollback["restored_sections"], ["cache"])
            self.assertTrue(rollback["reloaded_modules"])
            self.assertTrue(rollback["verification"]["ok"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), original)
            self.assertEqual(len(rollback["current_backups"]), 1)
            self.assertTrue(Path(rollback["current_backups"][0]["path"]).exists())
            events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
            event = [item for item in events if item.get("action") == "rollback"][-1]
            self.assertTrue(event["ok"])
            self.assertEqual(event["details"]["apply_id"], apply_result["apply_id"])
            self.assertEqual(event["details"]["restored_sections"], ["cache"])

    def test_policy_draft_rollback_restores_applied_file_when_reload_fails(self):
        async def failing_reload():
            raise RuntimeError("rollback reload unavailable")

        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"

            env = {
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-rollback-reload-fail",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])
                    apply_result = asyncio.run(apply_validated_policy_draft(
                        "cache-rollback-reload-fail",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                    ))
                    self.assertTrue(apply_result["ok"])
                    applied_text = cache_path.read_text(encoding="utf-8")

                    rollback = asyncio.run(rollback_policy_apply(
                        apply_result["apply_id"],
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=failing_reload,
                        event_source="test",
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(rollback["ok"])
            self.assertEqual(rollback["error"]["type"], "reload_failed")
            self.assertTrue(rollback["restored"])
            self.assertTrue(rollback["restore"]["ok"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), applied_text)
            self.assertNotEqual(cache_path.read_text(encoding="utf-8"), original)
            _assert_workbench_payload_is_metadata_only(self, rollback)

    def test_policy_draft_rollback_by_apply_id_missing_backup_fails_closed(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            config_dir.mkdir()
            cache_path = config_dir / "cache_rules.yaml"
            original = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            cache_path.write_text(original, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"

            env = {
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    stage = asyncio.run(stage_policy_draft(
                        {"semantic_cache": {"threshold": 0.91}},
                        section="cache",
                        draft_id="cache-rollback-missing",
                        workspace=workspace,
                    ))
                    self.assertTrue(stage["ok"])
                    apply_result = asyncio.run(apply_validated_policy_draft(
                        "cache-rollback-missing",
                        workspace=workspace,
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                    ))
                    self.assertTrue(apply_result["ok"])
                    Path(apply_result["backups"][0]["path"]).unlink()

                    rollback = asyncio.run(rollback_policy_apply(
                        apply_result["apply_id"],
                        config_dir=config_dir,
                        sections=["cache"],
                        reload_policy_state=reload_policy_modules,
                        event_source="test",
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(rollback["ok"])
            self.assertEqual(rollback["error"]["type"], "partial_backup_set")
            self.assertEqual(rollback["error"]["sections"], ["cache"])
            self.assertEqual(yaml.safe_load(cache_path.read_text(encoding="utf-8"))["semantic_cache"]["threshold"], 0.91)

    def test_policy_bundle_exports_manual_policy_source_and_file_status(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
    action:
      route_to: haiku
      reason: manual bundle export test
""",
                encoding="utf-8",
            )
            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                asyncio.run(reload_policy_modules())
                bundle = asyncio.run(build_policy_bundle())

                routing = bundle["policies"]["routing"]
                self.assertEqual(routing["policy_source"], "local-manual")
                self.assertEqual(routing["rule_path"], str(rules_path))
                self.assertFalse(routing["file"]["reload_required"])
                self.assertEqual(routing["file"]["loaded"]["path"], str(rules_path))
                self.assertEqual(routing["rules"][0]["action"]["reason"], "manual bundle export test")
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                asyncio.run(reload_policy_modules())

    def test_policy_file_status_marks_reload_required_after_file_change(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: initial
""",
                encoding="utf-8",
            )
            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                importlib.reload(router_module)
                first = asyncio.run(stats.stats_policies())
                self.assertEqual(first["routing"]["policy_source"], "local-manual")
                self.assertEqual(first["routing"]["rule_path"], str(rules_path))
                self.assertFalse(first["routing"]["file"]["reload_required"])
                self.assertFalse(first["summary"]["reload_required"])
                self.assertEqual(first["summary"]["reload_required_sections"], [])
                self.assertEqual(first["summary"]["manual_policy_count"], 1)
                self.assertTrue(first["routing"]["file"]["loaded"]["exists"])
                self.assertIsNotNone(first["routing"]["file"]["loaded"]["size"])
                self.assertIsNotNone(first["routing"]["file"]["loaded"]["mtime_ns"])

                rules_path.write_text(
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: changed
""",
                    encoding="utf-8",
                )

                changed = asyncio.run(stats.stats_policies())
                self.assertTrue(changed["routing"]["file"]["reload_required"])
                self.assertTrue(changed["summary"]["reload_required"])
                self.assertEqual(changed["summary"]["reload_required_sections"], ["routing"])
                self.assertNotEqual(
                    changed["routing"]["file"]["loaded"]["sha256"],
                    changed["routing"]["file"]["current"]["sha256"],
                )
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                importlib.reload(router_module)

    def test_policy_file_status_handles_missing_file_without_reload(self):
        missing = Path("/tmp/agentflow-policy-file-status-missing.yaml")
        loaded = policy_file_snapshot(missing)

        status = policy_file_status(missing, loaded_at=utc_now(), loaded_snapshot=loaded)

        self.assertFalse(status["loaded"]["exists"])
        self.assertFalse(status["current"]["exists"])
        self.assertFalse(status["reload_required"])

    def test_reload_policy_modules_applies_changed_routing_file(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
    action:
      route_to: haiku
      reason: initial reload test
""",
                encoding="utf-8",
            )

            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                first = asyncio.run(reload_policy_modules())
                self.assertEqual(first["schema"], "agentflow.policy_reload.v1")
                self.assertFalse(first["policies"]["routing"]["file"]["reload_required"])

                body = {
                    "model": router_module.SONNET_DEFAULT,
                    "messages": [{"role": "user", "content": "Say ok."}],
                }
                routed, meta = router_module.route_model(body)
                self.assertEqual(routed, router_module.HAIKU_DEFAULT)
                self.assertEqual(meta["reason"], "initial reload test")

                rules_path.write_text(
                    """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
    action:
      route_to: haiku
      reason: changed reload test
""",
                    encoding="utf-8",
                )
                stale = asyncio.run(stats.stats_policies())
                self.assertTrue(stale["routing"]["file"]["reload_required"])

                changed = asyncio.run(reload_policy_modules())
                self.assertFalse(changed["policies"]["routing"]["file"]["reload_required"])
                _routed, changed_meta = router_module.route_model(body)
                self.assertEqual(changed_meta["reason"], "changed reload test")
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                asyncio.run(reload_policy_modules())


if __name__ == "__main__":
    unittest.main()
