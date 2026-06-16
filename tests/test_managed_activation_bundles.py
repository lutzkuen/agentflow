from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from agentflow_proxy import cli
from agentflow_proxy.managed_activation_bundles import stage_managed_activation_bundle_sync


def _action(family: str, order: int, *, apply_after: list[str] | None = None) -> dict[str, object]:
    draft_id = f"{family}-draft-fixture"
    return {
        "schema": "agentflow.local_activation_policy_bundle_action.v1",
        "action_id": f"local-activation-policy-bundle-action:{family}:fixture",
        "draft_id": draft_id,
        "recommendation_id": f"{family}-recommendation-fixture",
        "activation_order": order,
        "apply_after": apply_after or [],
        "status": "review-required",
        "activation_mode": "widen-local-policy-review",
        "policy_source": "managed-recommended",
        "required_local_review": True,
        "managed_enforced": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "local_action_family": family,
        "local_executor_family": f"{family}-rules",
        "candidate_family": f"{family}-policy-rule",
        "policy_section": family,
        "target_local_rule_file": f"{family}-rules.yaml",
        "confidence": 0.92,
        "expected_savings": {
            "schema": "agentflow.local_activation_expected_savings.v1",
            "projected_savings_usd": 0.12,
            "projected_saved_tokens": 1234,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "required_coverage": {
            "schema": "agentflow.local_activation_required_coverage.v1",
            "observed_applied_count": 3,
            "observed_holdout_count": 2,
            "has_required_applied_coverage": True,
            "has_required_holdout_coverage": True,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "rollback_criteria": {
            "schema": "agentflow.local_activation_rollback_criteria.v1",
            "rollback_on_safety_stop": True,
            "local_rollback_required": True,
            "managed_enforced": False,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "keep_staged_criteria": {
            "schema": "agentflow.local_activation_keep_staged_criteria.v1",
            "require_no_safety_stop": True,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "risk_summary": {
            "schema": "agentflow.local_activation_risk_summary.v1",
            "required_local_review": True,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "candidate_bucket": {
            "policy_section": family,
            "target_local_rule_file": f"{family}-rules.yaml",
            "source_decision": "widen",
        },
        "local_policy_draft": {
            "schema": "agentflow.local_policy_rule_draft.v1",
            "status": "review-required",
            "policy_source": "managed-recommended",
            "managed_enforced": False,
            "feature_only": True,
            "locally_executed": True,
            "provider_forwarding": False,
            "server_content_processing": False,
            "policy_section": family,
            "local_action_family": family,
            "candidate_family": f"{family}-policy-rule",
            "rule_file": f"{family}-rules.yaml",
            "next_action": "widen",
            "candidate_bucket": {"policy_section": family, "source_decision": "widen"},
            "decision_provenance": {
                "schema": "agentflow.local_activation_outcome_decision_provenance.v1",
                "source_decision": "widen",
                "metadata_only": True,
                "aggregate_only": True,
                "feature_only": True,
            },
            "activation_order": order,
            "apply_after": apply_after or [],
            "confidence": 0.92,
            "review_note": "Managed draft only; local file-backed rules must be reviewed and applied locally.",
        },
        "provenance": {
            "schema": "agentflow.local_activation_outcome_policy_bundle_draft_provenance.v1",
            "issuer": "agentflow-server",
            "signing_status": "unsigned-development-draft",
            "managed_enforced": False,
        },
        "privacy_summary": {
            "feature_only": True,
            "metadata_only": True,
            "aggregate_only": True,
            "provider_forwarding": False,
            "server_content_processing": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "tool_payloads_included": False,
        },
    }


def _draft_for_action(action: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "agentflow.local_activation_outcome_policy_bundle_draft.v1",
        "status": "draft",
        "draft_id": action["draft_id"],
        "recommendation_id": action["recommendation_id"],
        "rank": action["activation_order"],
        "local_action_family": action["local_action_family"],
        "candidate_family": action["candidate_family"],
        "policy_section": action["policy_section"],
        "target_local_rule_file": action["target_local_rule_file"],
        "confidence": action["confidence"],
        "activation_order": action["activation_order"],
        "apply_after": action["apply_after"],
        "activation_mode": action["activation_mode"],
        "local_executor_family": action["local_executor_family"],
        "generated_at": "2026-06-16T04:00:00+00:00",
        "expires_at": "2999-01-01T00:00:00+00:00",
        "policy_source": "managed-recommended",
        "required_local_review": True,
        "managed_enforced": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "local_executor_compatibility": {
            "compatible": True,
            "minimum_local_client_version": "0.1.0",
            "requires_local_review": True,
            "managed_enforced": False,
        },
        "candidate_bucket": action["candidate_bucket"],
        "recommended_local_rule": action["local_policy_draft"],
        "expected_savings": action["expected_savings"],
        "required_coverage": action["required_coverage"],
        "rollback_criteria": action["rollback_criteria"],
        "keep_staged_criteria": action["keep_staged_criteria"],
        "risk_summary": action["risk_summary"],
        "evidence_summary": {
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "provenance": action["provenance"],
        "privacy_summary": action["privacy_summary"],
    }


def _bundle() -> dict[str, object]:
    cache = _action("cache", 1)
    crunch = _action("crunch", 2, apply_after=[str(cache["draft_id"])])
    return {
        "schema": "agentflow.local_activation_outcome_policy_bundle_drafts.v1",
        "bundle_id": "local-activation-policy-bundle:fixture",
        "status": "review-only",
        "generated_at": "2026-06-16T04:00:00+00:00",
        "summary": {
            "ordered_action_count": 2,
            "draft_count": 2,
            "omitted_count": 1,
            "ordered_action_families": ["cache", "crunch"],
            "expires_at": "2999-01-01T00:00:00+00:00",
            "policy_source": "managed-recommended",
            "required_local_review": True,
            "managed_enforced": False,
            "feature_only": True,
            "locally_executed": True,
            "provider_forwarding": False,
            "server_content_processing": False,
        },
        "local_actions": [cache, crunch],
        "drafts": [_draft_for_action(cache), _draft_for_action(crunch)],
        "omitted_actions": [
            {
                "schema": "agentflow.local_activation_outcome_policy_bundle_draft_omission.v1",
                "status": "omitted",
                "local_action_family": "routing",
                "reason_codes": ["local-activation-safety-stop"],
                "provider_forwarding": False,
                "server_content_processing": False,
            }
        ],
        "provenance": {
            "schema": "agentflow.local_activation_policy_bundle_provenance.v1",
            "issuer": "agentflow-server",
            "bundle_id": "local-activation-policy-bundle:fixture",
            "bundle_hash": "sha256:fixture",
            "signing_status": "unsigned-development-draft",
            "review_only": True,
            "managed_enforced": False,
            "metadata_only": True,
            "aggregate_only": True,
            "feature_only": True,
        },
        "privacy_summary": {
            "feature_only": True,
            "metadata_only": True,
            "aggregate_only": True,
            "provider_forwarding": False,
            "server_content_processing": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "tool_payloads_included": False,
            "absolute_paths_included": False,
        },
    }


class ManagedActivationBundleTests(unittest.TestCase):
    def test_import_stages_cache_and_crunch_policy_draft_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"

            result = stage_managed_activation_bundle_sync(
                _bundle(),
                workspace=workspace,
                config_dir=config_dir,
            )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["schema"], "agentflow.managed_activation_bundle_import.v1")
            self.assertEqual(result["status"], "staged")
            self.assertEqual(result["summary"]["staged_count"], 2)
            self.assertEqual(result["summary"]["skipped_count"], 0)
            self.assertEqual(result["summary"]["omitted_count"], 1)
            self.assertEqual(result["summary"]["target_local_rule_files"], ["cache_rules.yaml", "crunch_rules.yaml"])
            self.assertFalse(result["wrote_active_policy_files"])
            self.assertFalse(result["provider_calls_made"])
            self.assertFalse(result["managed_server_calls_made"])
            self.assertFalse((config_dir / "cache_rules.yaml").exists())
            self.assertFalse((config_dir / "crunch_rules.yaml").exists())

            staged_by_family = {item["local_action_family"]: item for item in result["staged"]}
            self.assertEqual(set(staged_by_family), {"cache", "crunch"})
            for family, row in staged_by_family.items():
                self.assertEqual(row["target_local_rule_file"], f"{family}_rules.yaml")
                self.assertTrue(row["stage"]["ok"], row)
                self.assertTrue(row["validation"]["validation"]["ok"], row["validation"])
                self.assertFalse(row["stage"]["wrote_active_policy_files"])
                section_path = Path(row["workspace"]) / "sections" / f"{family}_rules.yaml"
                self.assertTrue(section_path.exists())
                section_payload = yaml.safe_load(section_path.read_text(encoding="utf-8"))
                self.assertIn("managed_activation_drafts", section_payload)
                entry = section_payload["managed_activation_drafts"][0]
                self.assertEqual(entry["schema"], "agentflow.managed_activation_policy_draft_entry.v1")
                self.assertEqual(entry["policy_source"], "managed-recommended")
                self.assertEqual(entry["local_action_family"], family)
                self.assertEqual(entry["target_local_rule_file"], f"{family}_rules.yaml")
                self.assertFalse(entry["provider_forwarding"])
                self.assertFalse(entry["server_content_processing"])
                self.assertEqual(entry["local_policy_draft"]["status"], "review-required")

    def test_import_rejects_content_bearing_bundle_before_writing_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = _bundle()
            payload["prompt"] = "raw prompt must not be imported"
            workspace = Path(tmp) / "drafts"

            result = stage_managed_activation_bundle_sync(payload, workspace=workspace)

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["error"]["type"], "content-bearing-bundle-rejected")
            self.assertFalse(workspace.exists())
            self.assertIn("$.prompt", {error["path"] for error in result["error"]["errors"]})

    def test_cli_stages_managed_activation_bundle_from_stdin(self) -> None:
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.managed_activation_bundle_stage_cli(
                [
                    "--workspace",
                    str(Path(tmp) / "drafts"),
                    "--config-dir",
                    str(Path(tmp) / "config"),
                    "-",
                ],
                stdin=io.StringIO(json.dumps(_bundle())),
                stdout=stdout,
            )

            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 0, result)
            self.assertTrue(result["ok"])
            self.assertEqual(result["summary"]["staged_count"], 2)


if __name__ == "__main__":
    unittest.main()
