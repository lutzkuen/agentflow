import asyncio
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from agentflow_proxy import cache as cache_module
from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.optimization_promotion_actions import ACTION_SCHEMA, SCHEMA as PROMOTION_ACTIONS_SCHEMA, build_optimization_promotion_actions
from agentflow_proxy.optimization_promotion_canary import (
    apply_optimization_promotion_canaries,
    evaluate_promotion_canary_safety_stop,
    promotion_canary_decision,
)
from agentflow_proxy.optimization_rollout_review import (
    attach_optimization_rollout_provenance,
    review_optimization_rollout_actions,
    validate_optimization_rollout_bundle,
)
from agentflow_proxy.recommendations import queue_policy_event_feedback
from agentflow_proxy.store import stable_json


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "optimization_promotion_rollout_contract_v1.json"


class FakeQueuedPolicyEventStore:
    def __init__(self):
        self.rows = []

    def enqueue_managed_outcome_feedback(self, **kwargs):
        self.rows.append(kwargs)


class OptimizationPromotionRolloutFixtureTests(unittest.TestCase):
    def _fixture(self):
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def _signed(self, bundle):
        return attach_optimization_rollout_provenance(
            bundle,
            secret="fixture-secret",
            issuer="agentflow-server",
            server_id="managed-fixture",
            key_id="fixture-key",
            generated_at="2026-06-10T05:00:00+00:00",
        )

    def _assert_metadata_only(self, payload):
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw fixture prompt secret",
            "raw fixture response secret",
            "raw fixture provider body",
            "fixture-cache-key-secret",
            "fixture-request-id-secret",
            "fixture-session-id-secret",
            "/tmp/fixture-secret.py",
            "sk-fixture-secret",
        ):
            self.assertNotIn(forbidden, rendered)

    def _crunch_promotion_bundle(self, *, action_type="widen", raw_like=False, managed_enforced=False, include_hashes=True):
        pattern_hash = "sha256:" + ("a" * 64)
        local_update = {
            "kind": "yaml-rule-canary",
            "policy_source": "managed-enforced" if managed_enforced else "managed-recommended",
            "managed_enforced": managed_enforced,
            "required_local_review": True,
            "candidate_profile": "terminal_logs",
            "conditions": {
                "pattern_hashes": [pattern_hash] if include_hashes else [],
                "category": "tool-result",
                "min_repeated_count": 2,
                "keep_recent_matches": 1,
                "min_text_chars": 1000,
                "max_applications": 4,
            },
            "action": {
                "type": "shorten",
                "head_chars": 800,
                "tail_chars": 600,
                "max_replacement_chars": 1800,
                "marker": "[AgentFlow: terminal log sample preserved]",
            },
            "safety_stop": {
                "min_outcome_samples": 7,
                "rollback_threshold": 0.25,
            },
        }
        if raw_like:
            local_update["raw_prompt"] = "raw fixture prompt secret"
        canary_fraction = 0.0 if action_type == "rollback" else 0.2
        action = {
            "schema": ACTION_SCHEMA,
            "action_id": f"promotion-rollout-action:fixture-crunch-{action_type}",
            "status": "planned",
            "action_type": action_type,
            "verdict": "rollback" if action_type == "rollback" else "widen",
            "target_candidate_id": "fixture-crunch-candidate",
            "target_rule_id": "promotion-crunch-fixture",
            "action_family": "crunch",
            "optimization_family": "managed_pattern_candidate",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "policy_section": "crunch",
            "target_local_policy_section": "crunch.rules",
            "local_policy_update": local_update,
            "current_canary_fraction": 0.1,
            "canary_fraction": canary_fraction,
            "holdout_fraction": 0.1 if action_type != "rollback" else 0.0,
            "evidence_summary": {"eval_pass_count": 2, "eval_fail_count": 0, "reason_codes": ["promotion-thresholds-met"]},
            "rollback_metadata": {"preserve_previous_rule_required": True},
            "local_review": {"required": True, "apply_preview_command": "agentflow-optimization-promotion-canaries-apply --dry-run"},
            "privacy": {
                "metadata_only": True,
                "content_free": True,
                "raw_prompts_included": False,
                "raw_provider_bodies_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "raw_session_ids_included": False,
                "filesystem_paths_included": False,
            },
        }
        return {
            "schema": PROMOTION_ACTIONS_SCHEMA,
            "generated_at": "2026-06-10T06:00:00+00:00",
            "ok": True,
            "read_only": True,
            "wrote_local_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "summary": {"candidate_count": 1, "action_count": 1, "omitted_count": 0},
            "actions": [action],
            "omitted": [],
            "privacy": {"metadata_only": True, "content_free": True},
        }

    def _cache_promotion_bundle(
        self,
        *,
        action_type="widen",
        raw_like=False,
        managed_enforced=False,
        include_invalidation=True,
        has_tools=True,
        stream=False,
        canary_fraction=0.25,
    ):
        pattern_hash = "sha256:" + ("c" * 64)
        safe_invalidation = bool(include_invalidation)
        local_update = {
            "kind": "yaml-rule-canary",
            "policy_source": "managed-enforced" if managed_enforced else "managed-recommended",
            "managed_enforced": managed_enforced,
            "required_local_review": True,
            "candidate_profile": "replay-safe-exact-candidate",
            "conditions": {
                "pattern_hashes": [pattern_hash],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "tool-result" if has_tools else "short-completion",
                "workflow_phase": "tool-execution" if has_tools else "summary",
                "text_bucket": "2k_8k_chars" if has_tools else "lt_2k_chars",
                "token_bucket": "1k_4k_tokens" if has_tools else "lt_1k_tokens",
                "has_tools": has_tools,
                "stream": stream,
                "replayability_levels": ["local-exact-response"],
            },
            "action": {
                "type": "exact_cache_pattern",
                "allow_tool_calls": has_tools,
                "safe_invalidation": safe_invalidation,
                "safe_invalidation_evidence": safe_invalidation,
                "streaming": stream,
                "estimated_saved_cost_usd": 0.012,
            },
            "safety_stop": {
                "min_outcome_samples": 9,
                "rollback_threshold": 0.15,
            },
        }
        if raw_like:
            local_update["cache_key"] = "fixture-cache-key-secret"
        fraction = 0.0 if action_type == "rollback" else canary_fraction
        action = {
            "schema": ACTION_SCHEMA,
            "action_id": f"promotion-rollout-action:fixture-cache-{action_type}",
            "status": "planned",
            "action_type": action_type,
            "verdict": "rollback" if action_type == "rollback" else "widen",
            "target_candidate_id": "fixture-cache-candidate",
            "target_rule_id": "promotion-cache-fixture",
            "action_family": "cache",
            "optimization_family": "cache_replayability",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "policy_section": "cache",
            "target_local_policy_section": "cache.rules",
            "local_policy_update": local_update,
            "current_canary_fraction": 0.1,
            "canary_fraction": fraction,
            "holdout_fraction": 0.1 if action_type != "rollback" else 0.0,
            "evidence_summary": {"eval_pass_count": 2, "eval_fail_count": 0, "projected_savings_usd": 0.012},
            "rollback_metadata": {"preserve_previous_rule_required": True},
            "local_review": {"required": True, "apply_preview_command": "agentflow-optimization-promotion-canaries-apply --dry-run"},
            "privacy": {
                "metadata_only": True,
                "content_free": True,
                "raw_prompts_included": False,
                "raw_provider_bodies_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "raw_session_ids_included": False,
                "filesystem_paths_included": False,
                "cache_keys_included": False,
            },
        }
        return {
            "schema": PROMOTION_ACTIONS_SCHEMA,
            "generated_at": "2026-06-10T06:00:00+00:00",
            "ok": True,
            "read_only": True,
            "wrote_local_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "summary": {"candidate_count": 1, "action_count": 1, "omitted_count": 0},
            "actions": [action],
            "omitted": [],
            "privacy": {"metadata_only": True, "content_free": True},
        }

    def _write_policy_sentinels(self, tmp: str):
        crunch_path = Path(tmp) / "crunch_rules.yaml"
        routing_path = Path(tmp) / "routing_rules.yaml"
        cache_path = Path(tmp) / "cache_rules.yaml"
        crunch_path.write_text("enabled: true\nthreshold_chars: 24000\npattern_rules: []\n", encoding="utf-8")
        routing_path.write_text("rules:\n- name: keep-routing-sentinel\n", encoding="utf-8")
        cache_path.write_text("exact_cache:\n  enabled: true\npattern_rules: []\n", encoding="utf-8")
        return crunch_path, routing_path, cache_path

    def test_shared_managed_rollout_fixture_reviews_signed_actions_and_omissions(self):
        fixture = self._fixture()
        signed = self._signed(copy.deepcopy(fixture["managed_rollout_bundle"]))

        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "fixture-secret"}):
            result = review_optimization_rollout_actions(
                signed,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["provenance"]["status"], "verified")
        self.assertEqual(result["summary"]["accepted_action_count"], 2)
        self.assertEqual({row["policy_section"] for row in result["actions"]}, {"routing", "cache"})
        self.assertEqual(result["omitted_actions"][0]["reason"], "unsupported-local-policy-section")
        self.assertEqual(result["omitted_actions"][0]["target_candidate_id"], "fixture-unsupported-candidate")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self._assert_metadata_only(result)

    def test_shared_managed_rollout_fixture_rejects_raw_missing_omission_reason_and_missing_compatibility(self):
        fixture = self._fixture()

        raw_like = copy.deepcopy(fixture["managed_rollout_bundle"])
        raw_like["actions"][0]["raw_request"] = {"prompt": "raw fixture prompt secret"}
        raw_signed = self._signed(raw_like)
        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "fixture-secret"}):
            raw_result = validate_optimization_rollout_bundle(
                raw_signed,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )
        self.assertFalse(raw_result["ok"])
        self.assertIn("raw or local-identifier rollout payloads are not accepted", {error["message"] for error in raw_result["errors"]})

        missing_reason = copy.deepcopy(fixture["managed_rollout_bundle"])
        del missing_reason["omitted_actions"][0]["reason"]
        reason_signed = self._signed(missing_reason)
        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "fixture-secret"}):
            reason_result = validate_optimization_rollout_bundle(
                reason_signed,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )
        self.assertFalse(reason_result["ok"])
        self.assertIn("omitted action reason is required", {error["message"] for error in reason_result["errors"]})

        missing_compatibility = copy.deepcopy(fixture["managed_rollout_bundle"])
        del missing_compatibility["actions"][0]["local_executor_compatibility"]
        compat_signed = self._signed(missing_compatibility)
        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "fixture-secret"}):
            compat_result = validate_optimization_rollout_bundle(
                compat_signed,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )
        self.assertFalse(compat_result["ok"])
        self.assertIn("local executor compatibility contract is required", {error["message"] for error in compat_result["errors"]})

    def test_shared_promotion_fixture_drives_local_rollout_dry_run_and_raw_rejection(self):
        fixture = self._fixture()
        bundle = build_optimization_promotion_actions(
            fixture["promotion_report"],
            initial_canary_fraction=0.1,
            widen_step=0.25,
            holdout_fraction=0.1,
        )

        self.assertEqual(bundle["summary"]["action_count"], 2)
        self.assertEqual(bundle["summary"]["omitted_count"], 1)
        self.assertEqual(bundle["omitted"][0]["target_candidate_id"], "fixture-unsupported-candidate")
        self.assertEqual(bundle["omitted"][0]["reason"], "unsupported-local-policy-section")
        self._assert_metadata_only(bundle)

        with tempfile.TemporaryDirectory() as tmp:
            dry_run = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=True)
            self.assertTrue(dry_run["ok"])
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertEqual(dry_run["summary"]["planned_action_count"], 2)
            self.assertEqual(dry_run["summary"]["skipped_action_count"], 0)
            self.assertEqual({row["policy_section"] for row in dry_run["actions"]}, {"routing", "cache"})
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())

        unsafe_bundle = copy.deepcopy(bundle)
        unsafe_bundle["actions"][0]["raw_request"] = {"prompt": "raw fixture prompt secret"}
        with tempfile.TemporaryDirectory() as tmp:
            rejected = apply_optimization_promotion_canaries(unsafe_bundle, config_dir=tmp, dry_run=True)
        self.assertFalse(rejected["ok"])
        self.assertIn("raw or local-identifier promotion rollout payloads are not accepted", {error["message"] for error in rejected["errors"]})

    def test_crunch_promotion_canary_apply_dry_run_and_write_update_only_crunch_rules(self):
        bundle = self._crunch_promotion_bundle()

        with tempfile.TemporaryDirectory() as tmp:
            crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
            before_crunch = crunch_path.read_text(encoding="utf-8")
            before_routing = routing_path.read_text(encoding="utf-8")
            before_cache = cache_path.read_text(encoding="utf-8")

            dry_run = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=True)
            self.assertTrue(dry_run["ok"])
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertEqual(dry_run["summary"]["planned_action_count"], 1)
            self.assertEqual(dry_run["files"][0]["section"], "crunch")
            self.assertTrue(dry_run["files"][0]["changed"])
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), before_crunch)
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before_cache)

            applied = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)
            written = yaml.safe_load(crunch_path.read_text(encoding="utf-8"))

            self.assertTrue(applied["ok"])
            self.assertFalse(applied["dry_run"])
            self.assertTrue(applied["wrote_policy_files"])
            self.assertEqual(applied["files"][0]["section"], "crunch")
            self.assertIsNotNone(applied["files"][0]["backup_path"])
            self.assertEqual(len(list(Path(tmp).glob("crunch_rules.yaml.bak-*"))), 1)
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before_cache)

        rules = written["pattern_rules"]
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertEqual(rule["id"], "promotion-crunch-fixture")
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "managed-recommended")
        self.assertEqual(rule["candidate_id"], "fixture-crunch-candidate")
        self.assertEqual(rule["promotion_action_id"], "promotion-rollout-action:fixture-crunch-widen")
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:" + ("a" * 64)])
        self.assertEqual(rule["conditions"]["category"], "tool-result")
        self.assertEqual(rule["action"]["head_chars"], 800)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.2)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.1)
        self.assertEqual(rule["rollout"]["min_outcome_samples"], 7)
        self.assertEqual(rule["rollout_action"]["managed_enforced"], False)

    def test_crunch_promotion_canary_rollback_disables_only_matching_managed_rule(self):
        bundle = self._crunch_promotion_bundle(action_type="rollback", include_hashes=False)
        managed_rule = {
            "id": "promotion-crunch-fixture",
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": "fixture-crunch-candidate",
            "conditions": {"pattern_hashes": ["sha256:" + ("a" * 64)], "min_repeated_count": 2, "keep_recent_matches": 1},
            "action": {"type": "shorten", "head_chars": 800, "tail_chars": 600, "max_replacement_chars": 1800},
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "canary",
                "canary_enabled": True,
                "canary_fraction": 0.2,
                "holdout_fraction": 0.1,
                "canary_salt": "fixture",
                "canary_unit": "request_fingerprint",
            },
        }
        local_manual_rule = {
            "id": "local-manual-rule",
            "enabled": True,
            "policy_source": "local-manual",
            "candidate_id": "local-candidate",
            "conditions": {"pattern_hashes": ["sha256:" + ("b" * 64)], "min_repeated_count": 2},
            "action": {"type": "shorten", "head_chars": 1000, "tail_chars": 800},
        }

        with tempfile.TemporaryDirectory() as tmp:
            crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
            crunch_path.write_text(
                yaml.safe_dump(
                    {"enabled": True, "threshold_chars": 24000, "pattern_rules": [managed_rule, local_manual_rule]},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            before_routing = routing_path.read_text(encoding="utf-8")
            before_cache = cache_path.read_text(encoding="utf-8")

            result = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)
            written = yaml.safe_load(crunch_path.read_text(encoding="utf-8"))

            self.assertTrue(result["ok"])
            self.assertTrue(result["wrote_policy_files"])
            self.assertEqual(result["summary"]["planned_action_count"], 1)
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before_cache)

        self.assertEqual(len(written["pattern_rules"]), 2)
        rolled_back = next(rule for rule in written["pattern_rules"] if rule["id"] == "promotion-crunch-fixture")
        untouched = next(rule for rule in written["pattern_rules"] if rule["id"] == "local-manual-rule")
        self.assertFalse(rolled_back["enabled"])
        self.assertFalse(rolled_back["rollout"]["canary_enabled"])
        self.assertEqual(rolled_back["rollout"]["canary_fraction"], 0.0)
        self.assertEqual(rolled_back["rollout"]["holdout_fraction"], 0.0)
        self.assertEqual(rolled_back["conditions"]["pattern_hashes"], ["sha256:" + ("a" * 64)])
        self.assertTrue(untouched["enabled"])
        self.assertEqual(untouched["policy_source"], "local-manual")

    def test_crunch_promotion_canary_rejects_raw_payload_and_managed_enforced_before_writing(self):
        for bundle in (
            self._crunch_promotion_bundle(raw_like=True),
            self._crunch_promotion_bundle(managed_enforced=True),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
                before = {
                    "crunch": crunch_path.read_text(encoding="utf-8"),
                    "routing": routing_path.read_text(encoding="utf-8"),
                    "cache": cache_path.read_text(encoding="utf-8"),
                }

                result = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)

                self.assertFalse(result["ok"])
                self.assertFalse(result["wrote_policy_files"])
                self.assertEqual(crunch_path.read_text(encoding="utf-8"), before["crunch"])
                self.assertEqual(routing_path.read_text(encoding="utf-8"), before["routing"])
                self.assertEqual(cache_path.read_text(encoding="utf-8"), before["cache"])

    def test_cache_promotion_canary_apply_dry_run_write_and_lookup_metadata(self):
        bundle = self._cache_promotion_bundle(has_tools=False, canary_fraction=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
            before_crunch = crunch_path.read_text(encoding="utf-8")
            before_routing = routing_path.read_text(encoding="utf-8")
            before_cache = cache_path.read_text(encoding="utf-8")

            dry_run = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=True)
            self.assertTrue(dry_run["ok"])
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertEqual(dry_run["summary"]["planned_action_count"], 1)
            self.assertEqual(dry_run["files"][0]["section"], "cache")
            self.assertTrue(dry_run["files"][0]["changed"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before_cache)

            applied = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)
            written = yaml.safe_load(cache_path.read_text(encoding="utf-8"))

            self.assertTrue(applied["ok"])
            self.assertTrue(applied["wrote_policy_files"])
            self.assertEqual(applied["files"][0]["section"], "cache")
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), before_crunch)
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)

        rules = written["pattern_rules"]
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        pattern_hash = "sha256:" + ("c" * 64)
        self.assertEqual(rule["id"], "promotion-cache-fixture")
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "managed-recommended")
        self.assertEqual(rule["candidate_id"], "fixture-cache-candidate")
        self.assertEqual(rule["conditions"]["pattern_hashes"], [pattern_hash])
        self.assertEqual(rule["conditions"]["replayability_levels"], ["local-exact-response"])
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertEqual(rule["action"]["type"], "exact_cache_pattern")
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.0)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.1)
        self.assertEqual(rule["rollout"]["min_outcome_samples"], 9)

        normalized = cache_module.normalize_cache_pattern_rules(rules)
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = tuple(normalized)
            can_exact, can_semantic, meta = cache_module.cache_lookup_meta(
                has_tool_blocks=False,
                pattern_features={
                    "pattern_hashes": [pattern_hash],
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "short-completion",
                    "workflow_phase": "summary",
                    "text_bucket": "lt_2k_chars",
                    "token_bucket": "lt_1k_tokens",
                    "has_tools": False,
                    "stream": False,
                    "request_fingerprint": "fixture-fingerprint",
                    "raw_pattern_strings_included": False,
                },
            )
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

        self.assertTrue(can_exact)
        self.assertFalse(can_semantic)
        holdout = meta["pattern_rules"]["skip_reasons"][-1]
        self.assertEqual(holdout["reason"], "canary_holdout")
        self.assertEqual(holdout["candidate_id"], "fixture-cache-candidate")
        self.assertEqual(holdout["canary"]["status"], "holdout")
        self.assertFalse(holdout["canary"]["raw_pattern_strings_included"])
        self._assert_metadata_only(meta)

    def test_cache_promotion_canary_rejects_tool_rule_without_invalidation_evidence(self):
        bundle = self._cache_promotion_bundle(include_invalidation=False, has_tools=True)

        with tempfile.TemporaryDirectory() as tmp:
            crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
            before = {
                "crunch": crunch_path.read_text(encoding="utf-8"),
                "routing": routing_path.read_text(encoding="utf-8"),
                "cache": cache_path.read_text(encoding="utf-8"),
            }

            result = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)

            self.assertFalse(result["ok"])
            self.assertFalse(result["wrote_policy_files"])
            self.assertEqual(result["actions"][0]["reason"], "missing-safe-invalidation-evidence")
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), before["crunch"])
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before["routing"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before["cache"])

    def test_cache_promotion_canary_rollback_disables_only_matching_managed_rule(self):
        bundle = self._cache_promotion_bundle(action_type="rollback")
        managed_rule = {
            "id": "promotion-cache-fixture",
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": "fixture-cache-candidate",
            "conditions": {
                "pattern_hashes": ["sha256:" + ("c" * 64)],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "tool-result",
                "has_tools": True,
                "stream": False,
                "replayability_levels": ["local-exact-response"],
            },
            "action": {
                "type": "exact_cache_pattern",
                "allow_tool_calls": True,
                "safe_invalidation": True,
                "safe_invalidation_evidence": True,
                "streaming": False,
            },
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "canary",
                "canary_enabled": True,
                "canary_fraction": 0.25,
                "holdout_fraction": 0.1,
                "canary_salt": "fixture",
                "canary_unit": "request_fingerprint",
            },
        }
        local_manual_rule = {
            "id": "local-manual-cache-rule",
            "enabled": True,
            "policy_source": "local-manual",
            "candidate_id": "local-cache-candidate",
            "conditions": {"pattern_hashes": ["sha256:" + ("d" * 64)], "has_tools": False},
            "action": {"type": "exact_cache_pattern", "allow_tool_calls": False},
        }

        with tempfile.TemporaryDirectory() as tmp:
            crunch_path, routing_path, cache_path = self._write_policy_sentinels(tmp)
            cache_path.write_text(
                yaml.safe_dump(
                    {"exact_cache": {"enabled": True, "cache_tool_calls": False}, "pattern_rules": [managed_rule, local_manual_rule]},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            before_crunch = crunch_path.read_text(encoding="utf-8")
            before_routing = routing_path.read_text(encoding="utf-8")

            result = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)
            written = yaml.safe_load(cache_path.read_text(encoding="utf-8"))

            self.assertTrue(result["ok"])
            self.assertTrue(result["wrote_policy_files"])
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), before_crunch)
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)

        rolled_back = next(rule for rule in written["pattern_rules"] if rule["id"] == "promotion-cache-fixture")
        untouched = next(rule for rule in written["pattern_rules"] if rule["id"] == "local-manual-cache-rule")
        self.assertFalse(rolled_back["enabled"])
        self.assertFalse(rolled_back["rollout"]["canary_enabled"])
        self.assertEqual(rolled_back["rollout"]["canary_fraction"], 0.0)
        self.assertEqual(rolled_back["rollout"]["holdout_fraction"], 0.0)
        self.assertTrue(untouched["enabled"])
        self.assertEqual(untouched["policy_source"], "local-manual")

    def test_shared_canary_safety_stop_and_lifecycle_feedback_are_metadata_only(self):
        fixture = self._fixture()
        bundle = build_optimization_promotion_actions(
            fixture["promotion_report"],
            initial_canary_fraction=0.1,
            widen_step=0.25,
            holdout_fraction=0.1,
        )
        action = next(row for row in bundle["actions"] if row["policy_section"] == "routing")
        safety = evaluate_promotion_canary_safety_stop(
            action,
            fixture["safety_stop_records"],
            thresholds={"min_samples": 2, "max_error_rate": 0.5, "max_5xx_rate": 0.0, "max_unsupported_model_errors": 0},
        )
        decision = promotion_canary_decision(action, fixture["canary_metadata"], safety_stop=safety)

        self.assertTrue(safety["tripped"])
        self.assertIn("unsupported-model-errors", safety["reason_codes"])
        self.assertEqual(decision["status"], "safety_stopped")
        self.assertEqual(decision["reason"], "local-canary-safety-stop")
        self.assertFalse(decision["raw_content_included"])
        self._assert_metadata_only(decision)

        event = copy.deepcopy(fixture["lifecycle_feedback_event"])
        event["metadata"]["safety_stop_reason_counts"] = {reason: 1 for reason in safety["reason_codes"]}
        self.assertEqual(managed_egress_violations(event), [])
        store = FakeQueuedPolicyEventStore()
        with patch.dict(
            "os.environ",
            {
                "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
            },
        ):
            meta = asyncio.run(queue_policy_event_feedback(
                store,
                event,
                source_surface="optimization_promotion_lifecycle",
            ))

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(store.rows[0]["source_surface"], "optimization_promotion_lifecycle")
        payload = json.loads(store.rows[0]["payload_json"])
        self.assertEqual(payload["metadata"]["lifecycle_kind"], "optimization_promotion_rollout")
        self.assertFalse(payload["metadata"]["privacy"]["raw_prompts_included"])
        self._assert_metadata_only(payload)
        self.assertNotIn("payload_json", stable_json(meta))


if __name__ == "__main__":
    unittest.main()
