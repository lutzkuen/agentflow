from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agentflow_proxy import cli
from agentflow_proxy.optimization_rollout_review import attach_optimization_rollout_provenance
from agentflow_proxy.scaffold_rollout_review import (
    SCAFFOLD_CANARY_POLICY_FILE,
    apply_scaffold_rollout_actions,
    review_scaffold_rollout_actions,
)
from agentflow_proxy.store import stable_json


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

    def test_apply_cli_writes_scaffold_canary_overlay_by_default(self) -> None:
        signed = self._signed(self._bundle(canary_fraction=0.5))
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

    def test_crunch_loads_scaffold_overlay_and_applies_metadata_only_rule(self) -> None:
        signed = self._signed(self._bundle(canary_fraction=1.0))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "scaffold-review-secret"}):
            applied = apply_scaffold_rollout_actions(signed, config_dir=tmp)
            self.assertTrue(applied["ok"])
            overlay_path = str(Path(tmp) / SCAFFOLD_CANARY_POLICY_FILE)
            with patch.dict(os.environ, {"AGENTFLOW_SCAFFOLD_CANARY_POLICY": overlay_path}):
                import agentflow_proxy.crunch as crunch

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

            import agentflow_proxy.crunch as crunch

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
