from __future__ import annotations

import copy
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tokenclaw import cli
from tokenclaw.optimization_rollout_review import attach_optimization_rollout_provenance
from tokenclaw.scaffold_rollout_review import (
    SCAFFOLD_CANARY_POLICY_FILE,
    SCAFFOLD_LOCAL_CRUNCH_RULES_FILE,
    apply_scaffold_rollout_actions,
    review_scaffold_rollout_actions,
)
from tokenclaw.store import stable_json


class ScaffoldRolloutReviewTests(unittest.TestCase):
    def _bundle(self, *, canary_fraction: float = 1.0) -> dict:
        return {
            "schema": "agentflow.optimization_rollout_actions.v1",
            "generated_at": "2026-06-12T00:00:00+00:00",
            "expires_at": "2099-06-12T00:00:00+00:00",
            "summary": {
                "candidate_count": 1,
                "action_count": 1,
                "omitted_count": 0,
                "managed_enforced": False,
                "required_local_review": True,
                "provider_forwarding": False,
                "server_content_processing": False,
            },
            "local_executor_compatibility": {
                "minimum_local_client_version": "0.1.0",
                "compatible": True,
                "supported_local_action_families": ["crunch"],
                "local_review_required": True,
            },
            "actions": [
                {
                    "schema": "agentflow.optimization_rollout_action.v1",
                    "action_id": "rollout-action:repeated-scaffold:test",
                    "action_type": "review-local-repeated-scaffold-crunch-rule",
                    "target_candidate_id": "repeated-scaffold-candidate:test",
                    "action_family": "crunch",
                    "candidate_family": "repeated-scaffold-crunch-policy-rule",
                    "policy_section": "crunch",
                    "policy_source": "managed-recommended",
                    "source_surface": "anthropic_messages",
                    "provider_endpoint": "messages",
                    "confidence": 0.91,
                    "generated_at": "2026-06-12T00:00:00+00:00",
                    "expires_at": "2099-06-12T00:00:00+00:00",
                    "required_local_review": True,
                    "managed_enforced": False,
                    "local_executor_compatibility": {
                        "minimum_local_client_version": "0.1.0",
                        "compatible": True,
                        "supported_local_action_families": ["crunch"],
                        "local_review_required": True,
                        "requires_repeated_scaffold_support": True,
                    },
                    "evidence_summary": {
                        "rollout_gate": {"status": "pass"},
                        "repeated_scaffold": {
                            "applied_count": 4,
                            "holdout_count": 4,
                            "public_labels": ["provider-message-scaffold"],
                        },
                    },
                    "action": {
                        "status": "review-local-repeated-scaffold-crunch-rule",
                        "review_only": True,
                        "managed_enforced": False,
                        "locally_executed": False,
                        "local_policy_file": "crunch_rules.yaml",
                        "canary_fraction": canary_fraction,
                        "holdout_fraction": round(1.0 - canary_fraction, 6),
                        "thresholds": {
                            "min_samples": 4,
                            "max_error_rate": 0.05,
                            "max_retry_rate": 0.05,
                            "max_latency_delta_ms": 2000,
                            "min_repeated_blocks": 2,
                            "min_duplicate_blocks": 2,
                            "min_text_chars": 8000,
                        },
                        "proposed_edit": {
                            "id": "managed-repeated-scaffold-test",
                            "enabled": True,
                            "policy_source": "managed-recommended",
                            "candidate_id": "repeated-scaffold-candidate:test",
                            "description": "Review bounded provider-message scaffold crunching.",
                            "conditions": {
                                "source_surface": "anthropic_messages",
                                "app_family": "claude_code",
                                "phase": "tool-execution",
                                "category": "long-context",
                                "requested_model": "claude-sonnet-4-6",
                                "has_tools": False,
                                "uses_thinking": False,
                                "public_labels": ["provider-message-scaffold"],
                            },
                            "action": {
                                "type": "repeated_provider_message_scaffold_crunch",
                                "mode": "bounded_head_tail_replacement",
                                "review_only": True,
                                "replacement_notice": "[repeated provider-message scaffold removed locally]",
                                "min_repeated_blocks": 2,
                                "min_duplicate_blocks": 2,
                                "max_replacements_per_request": 4,
                                "canary_fraction": canary_fraction,
                                "holdout_fraction": round(1.0 - canary_fraction, 6),
                            },
                        },
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "feature_only": True,
                        "raw_payloads_returned": False,
                        "raw_prompts_returned": False,
                        "raw_responses_returned": False,
                        "provider_bodies_returned": False,
                        "request_ids_returned": False,
                        "tenant_ids_returned": False,
                        "cache_keys_returned": False,
                        "file_paths_returned": False,
                        "provider_forwarding": False,
                        "server_content_processing": False,
                        "managed_enforced": False,
                    },
                }
            ],
            "omitted_actions": [],
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
                "provider_forwarding": False,
                "server_content_processing": False,
                "managed_enforced": False,
            },
        }

    def _signed(self, bundle: dict) -> dict:
        return attach_optimization_rollout_provenance(
            bundle,
            secret="scaffold-review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="scaffold-review",
            generated_at="2026-06-12T00:00:00+00:00",
        )

    def _decision_bundle(self, decision: str, *, omitted: bool = False, canary_fraction: float = 0.5) -> dict:
        bundle = self._bundle(canary_fraction=canary_fraction)
        action = copy.deepcopy(bundle["actions"][0])
        action["action_type"] = decision
        action["action"]["next_action"] = decision
        action["action"]["status"] = "review-local-repeated-scaffold-crunch-rule"
        action["evidence_summary"]["rollout_gate"]["next_action"] = decision
        action["evidence_summary"]["rollout_decision"] = {
            "schema": "agentflow.repeated_scaffold_rollout_decision.v1",
            "next_action": decision,
            "reason_codes": [] if decision in {"widen", "promote"} else [f"fixture-{decision}"],
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "request_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
            },
        }
        if omitted:
            omitted_action = {
                "schema": "agentflow.repeated_scaffold_rollout_omitted_action.v1",
                "target_candidate_id": action["target_candidate_id"],
                "action_id": action["action_id"],
                "action_family": "crunch",
                "candidate_family": "repeated-scaffold-crunch-policy-rule",
                "source_surface": "anthropic_messages",
                "policy_section": "crunch",
                "reason": f"fixture-{decision}",
                "reason_codes": [f"fixture-{decision}"],
                "next_action": decision,
                "noop_action": {
                    "status": decision,
                    "local_action": "crunch",
                    "locally_executable": False,
                    "locally_executed": False,
                    "requires_local_review": True,
                    "managed_enforced": False,
                    "provider_forwarding": False,
                    "server_content_processing": False,
                },
                "evidence_summary": action["evidence_summary"],
                "privacy_summary": action["privacy_summary"],
            }
            bundle["actions"] = []
            bundle["omitted_actions"] = [omitted_action]
            bundle["summary"]["action_count"] = 0
            bundle["summary"]["omitted_count"] = 1
        else:
            bundle["actions"] = [action]
            bundle["omitted_actions"] = []
        return bundle

    def test_review_accepts_signed_repeated_scaffold_action_without_raw_payloads(self) -> None:
        signed = self._signed(self._bundle())
        with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            result = review_scaffold_rollout_actions(signed)

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema"], "agentflow.scaffold_rollout_actions_fetch_review.v1")
        self.assertEqual(result["accepted_action_count"], 1)
        self.assertEqual(result["provenance"]["status"], "verified")
        rule = result["actions"][0]["proposed_rule"]
        self.assertTrue(rule["match_any_repeated"])
        self.assertEqual(rule["policy_source"], "managed-recommended")
        self.assertEqual(rule["rollout"]["canary_fraction"], 1.0)
        rendered = stable_json(result)
        self.assertNotIn("raw prompt", rendered)
        self.assertFalse(result["privacy"]["raw_provider_bodies_included"])

    def test_review_rejects_raw_like_scaffold_rollout_decoys_without_leaking_values(self) -> None:
        bundle = self._bundle()
        action = bundle["actions"][0]
        action["raw_request"] = {"messages": [{"content": "raw action request must not leak"}]}
        action["action"]["proposed_edit"]["messages"] = [{"content": "raw proposed messages must not leak"}]
        action["action"]["proposed_edit"]["provider_body"] = {"content": "raw provider body must not leak"}
        action["action"]["proposed_edit"]["request_id"] = "raw-request-id-must-not-leak"
        action["action"]["proposed_edit"]["session_id"] = "raw-session-id-must-not-leak"
        action["action"]["proposed_edit"]["cache_key"] = "raw-cache-key-must-not-leak"
        action["action"]["proposed_edit"]["file_path"] = "/tmp/raw-scaffold-secret.py"
        action["privacy_summary"]["raw_prompts_returned"] = True
        bundle["privacy_summary"]["provider_bodies_returned"] = True
        signed = self._signed(bundle)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            review = review_scaffold_rollout_actions(signed)
            applied = apply_scaffold_rollout_actions(signed, config_dir=tmp, dry_run=False)

            self.assertFalse(review["ok"])
            self.assertFalse(applied["ok"])
            self.assertFalse(applied["wrote_policy_files"])
            self.assertFalse((Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE).exists())
            self.assertFalse((Path(tmp) / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE).exists())

        error_paths = {error["path"] for error in review["errors"]}
        self.assertIn("$.privacy_summary.provider_bodies_returned", error_paths)
        self.assertTrue(any("raw_request" in path or "messages" in path for path in error_paths))
        rendered = stable_json(review) + stable_json(applied)
        for forbidden in (
            "raw action request must not leak",
            "raw proposed messages must not leak",
            "raw provider body must not leak",
            "raw-request-id-must-not-leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "/tmp/raw-scaffold-secret.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(review["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(review["privacy"]["request_ids_included"])
        self.assertFalse(review["privacy"]["cache_keys_included"])

    def test_invalid_or_incompatible_scaffold_rollout_bundles_fail_closed(self) -> None:
        expired = self._bundle()
        expired["expires_at"] = "2000-01-01T00:00:00+00:00"
        expired["actions"][0]["expires_at"] = "2000-01-01T00:00:00+00:00"

        unsupported = self._bundle()
        unsupported["local_executor_compatibility"]["supported_local_action_families"] = ["routing"]
        unsupported["actions"][0]["local_executor_compatibility"]["supported_local_action_families"] = ["routing"]

        cases = {
            "expired": self._signed(expired),
            "unsigned": self._bundle(),
            "capability-mismatch": self._signed(unsupported),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            for name, bundle in cases.items():
                with self.subTest(name=name):
                    result = apply_scaffold_rollout_actions(bundle, config_dir=tmp, dry_run=False)

                    self.assertFalse(result["ok"])
                    self.assertFalse(result["wrote_policy_files"])
                    self.assertEqual(result["accepted_action_count"], 0)
                    self.assertEqual(result["error"]["type"], "validation_failed")
            self.assertFalse((Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE).exists())
            self.assertFalse((Path(tmp) / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE).exists())

    def test_stale_and_missing_holdout_lifecycle_omissions_preserve_pass_through_policy(self) -> None:
        cases = {
            "stale-evidence": self._decision_bundle("hold", omitted=True),
            "missing-holdout": self._decision_bundle("hold", omitted=True),
        }
        cases["stale-evidence"]["omitted_actions"][0]["reason_codes"] = ["stale-evidence"]
        cases["stale-evidence"]["omitted_actions"][0]["reason"] = "stale-evidence"
        cases["missing-holdout"]["omitted_actions"][0]["reason_codes"] = ["insufficient-holdout-samples"]
        cases["missing-holdout"]["omitted_actions"][0]["reason"] = "insufficient-holdout-samples"

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            for name, bundle in cases.items():
                with self.subTest(name=name):
                    result = apply_scaffold_rollout_actions(self._signed(bundle), config_dir=tmp, dry_run=False)

                    self.assertTrue(result["ok"])
                    self.assertFalse(result["wrote_policy_files"])
                    self.assertEqual(result["accepted_action_count"], 1)
                    self.assertEqual(result["files"][0]["reason"], "hold-no-local-policy-change")
            self.assertFalse((Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE).exists())
            self.assertFalse((Path(tmp) / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE).exists())

    def test_apply_cli_writes_scaffold_canary_overlay_by_default(self) -> None:
        signed = self._signed(self._decision_bundle("widen", canary_fraction=0.5))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            stdout = io.StringIO()
            code = cli.scaffold_rollout_actions_apply_cli(
                ["--config-dir", tmp, "--pretty", "-"],
                stdin=io.StringIO(json.dumps(signed)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["dry_run"])
            self.assertTrue(payload["wrote_policy_files"])
            path = Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE
            self.assertTrue(path.exists())
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            scaffold = data["repeated_provider_scaffolding"]
            self.assertTrue(scaffold["enabled"])
            self.assertEqual(scaffold["rules"][0]["candidate_id"], "repeated-scaffold-candidate:test")
            self.assertEqual(scaffold["rules"][0]["rollout"]["canary_fraction"], 0.5)
            rendered = yaml.safe_dump(data, sort_keys=True)
            for forbidden in (
                "raw_prompt",
                "messages:",
                "content:",
                "provider_body",
                "request_id",
                "session_id",
                "cache_key",
                "file_path",
                "pattern_hash",
            ):
                self.assertNotIn(forbidden, rendered)

    def test_review_accepts_widen_promote_hold_rollback_and_suppress_actions(self) -> None:
        cases = {
            "widen": False,
            "promote": False,
            "hold": True,
            "rollback": True,
            "suppress": True,
        }
        with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            for decision, omitted in cases.items():
                with self.subTest(decision=decision):
                    result = review_scaffold_rollout_actions(self._signed(self._decision_bundle(decision, omitted=omitted)))

                    self.assertTrue(result["ok"])
                    self.assertEqual(result["accepted_action_count"], 1)
                    self.assertEqual(result["actions"][0]["decision"], decision)
                    rendered = stable_json(result)
                    self.assertNotIn("raw prompt", rendered)
                    self.assertFalse(result["privacy"]["raw_provider_bodies_included"])

    def test_dry_run_outputs_are_metadata_only_for_lifecycle_decisions(self) -> None:
        cases = {
            "widen": False,
            "promote": False,
            "hold": True,
            "rollback": True,
            "suppress": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            for decision, omitted in cases.items():
                with self.subTest(decision=decision):
                    result = apply_scaffold_rollout_actions(
                        self._signed(self._decision_bundle(decision, omitted=omitted)),
                        config_dir=tmp,
                        dry_run=True,
                    )

                    self.assertTrue(result["ok"])
                    self.assertFalse(result["wrote_policy_files"])
                    self.assertFalse((Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE).exists())
                    self.assertFalse((Path(tmp) / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE).exists())
                    self.assertNotIn("raw prompt", stable_json(result))

    def test_promote_writes_durable_crunch_rule_with_backup(self) -> None:
        signed = self._signed(self._decision_bundle("promote", canary_fraction=1.0))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            crunch_path = Path(tmp) / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE
            crunch_path.write_text(
                yaml.safe_dump(
                    {
                        "repeated_provider_scaffolding": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "local-manual-unrelated",
                                    "candidate_id": "manual-candidate",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "match_any_repeated": True,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = apply_scaffold_rollout_actions(signed, config_dir=tmp, dry_run=False)

            self.assertTrue(result["ok"])
            self.assertTrue(result["wrote_policy_files"])
            crunch_file = next(file for file in result["files"] if file["path"].endswith(SCAFFOLD_LOCAL_CRUNCH_RULES_FILE))
            self.assertTrue(crunch_file["changed"])
            self.assertIsNotNone(crunch_file["backup_path"])
            self.assertTrue(Path(crunch_file["backup_path"]).exists())
            self.assertFalse((Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE).exists())
            data = yaml.safe_load(crunch_path.read_text(encoding="utf-8"))
            rules = data["repeated_provider_scaffolding"]["rules"]
            promoted = next(rule for rule in rules if rule["candidate_id"] == "repeated-scaffold-candidate:test")
            manual = next(rule for rule in rules if rule["id"] == "local-manual-unrelated")
            self.assertTrue(promoted["enabled"])
            self.assertEqual(promoted["policy_source"], "managed-recommended")
            self.assertFalse(promoted["rollout"]["canary_enabled"])
            self.assertEqual(promoted["rollout"]["canary_fraction"], 1.0)
            self.assertEqual(promoted["rollout_action"]["next_action"], "promote")
            self.assertTrue(manual["enabled"])
            self.assertEqual(manual["policy_source"], "local-manual")

    def test_rollback_disables_only_targeted_repeated_scaffold_rule(self) -> None:
        signed = self._signed(self._decision_bundle("rollback", omitted=True))
        target_rule = {
            "id": "managed-repeated-scaffold-test",
            "candidate_id": "repeated-scaffold-candidate:test",
            "enabled": True,
            "policy_source": "managed-recommended",
            "match_any_repeated": True,
            "rollout": {"canary_enabled": True, "canary_fraction": 0.5},
        }
        other_rule = {
            "id": "local-manual-unrelated",
            "candidate_id": "manual-candidate",
            "enabled": True,
            "policy_source": "local-manual",
            "match_any_repeated": True,
            "rollout": {"canary_enabled": True, "canary_fraction": 0.5},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            for filename in (SCAFFOLD_CANARY_POLICY_FILE, SCAFFOLD_LOCAL_CRUNCH_RULES_FILE):
                (Path(tmp) / filename).write_text(
                    yaml.safe_dump(
                        {
                            "repeated_provider_scaffolding": {
                                "enabled": True,
                                "rules": [copy.deepcopy(target_rule), copy.deepcopy(other_rule)],
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            result = apply_scaffold_rollout_actions(signed, config_dir=tmp, dry_run=False)

            self.assertTrue(result["ok"])
            self.assertTrue(result["wrote_policy_files"])
            changed_files = [file for file in result["files"] if file["changed"]]
            self.assertEqual(len(changed_files), 2)
            self.assertTrue(all(Path(file["backup_path"]).exists() for file in changed_files))
            for filename in (SCAFFOLD_CANARY_POLICY_FILE, SCAFFOLD_LOCAL_CRUNCH_RULES_FILE):
                data = yaml.safe_load((Path(tmp) / filename).read_text(encoding="utf-8"))
                rules = data["repeated_provider_scaffolding"]["rules"]
                rolled_back = next(rule for rule in rules if rule["candidate_id"] == "repeated-scaffold-candidate:test")
                manual = next(rule for rule in rules if rule["id"] == "local-manual-unrelated")
                self.assertFalse(rolled_back["enabled"])
                self.assertFalse(rolled_back["rollout"]["canary_enabled"])
                self.assertEqual(rolled_back["rollout"]["canary_fraction"], 0.0)
                self.assertEqual(rolled_back["rollout_action"]["next_action"], "rollback")
                self.assertEqual(rolled_back["rollout_action"]["rollback_reason_codes"], ["fixture-rollback"])
                self.assertTrue(manual["enabled"])
                self.assertEqual(manual["policy_source"], "local-manual")

    def test_crunch_loads_scaffold_overlay_and_applies_metadata_only_rule(self) -> None:
        signed = self._signed(self._bundle(canary_fraction=1.0))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            applied = apply_scaffold_rollout_actions(signed, config_dir=tmp)
            self.assertTrue(applied["ok"])
            overlay_path = str(Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE)
            with patch.dict(os.environ, {"AGENTFLOW_SCAFFOLD_CANARY_POLICY": overlay_path}):
                import tokenclaw.crunch as crunch

                reloaded = importlib.reload(crunch)
                repeated = "Provider message scaffold " + ("stable framing " * 800)
                body = {
                    "model": "claude-sonnet-4-6",
                    "messages": [
                        {"role": "user", "content": repeated},
                        {"role": "assistant", "content": "ack"},
                        {"role": "user", "content": repeated},
                        {"role": "assistant", "content": "ack"},
                        {"role": "user", "content": repeated},
                        {"role": "assistant", "content": "ack"},
                        {"role": "user", "content": repeated},
                    ],
                }
                changed, meta = reloaded.crunch_body(body)

            import tokenclaw.crunch as crunch

            importlib.reload(crunch)

        provider_meta = meta["repeated_provider_scaffolding"]
        self.assertTrue(provider_meta["enabled"])
        self.assertTrue(provider_meta["changed"])
        self.assertGreater(provider_meta["applied_count"], 0)
        self.assertIn("scaffold_canary_policy.yaml", provider_meta["rule_path"])
        self.assertEqual(provider_meta["rules"][0]["policy_source"], "managed-recommended")
        self.assertTrue(provider_meta["rules"][0]["match_any_repeated"])
        self.assertTrue(changed)
        self.assertFalse(provider_meta["raw_hashes_included"])
