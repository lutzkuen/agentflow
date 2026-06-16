from __future__ import annotations

import io
import json
import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from agentflow_proxy import cli
from agentflow_proxy.managed_activation_apply import apply_staged_managed_activation_bundle
from agentflow_proxy.managed_activation_bundles import stage_managed_activation_bundle_sync
from agentflow_proxy.policy_workbench import rollback_policy_apply


def _action(family: str, order: int, *, apply_after: list[str] | None = None) -> dict[str, object]:
    draft_id = f"{family}-draft-fixture"
    if family == "cache":
        target_policy_section = "cache.pattern_rules"
        local_policy_patch = {
            "pattern_rules": [
                {
                    "id": "managed-cache-replay-fixture",
                    "enabled": True,
                    "policy_source": "managed-recommended",
                    "candidate_id": "managed-cache-replay-fixture",
                    "conditions": {
                        "pattern_hashes": ["sha256:fixture-cache-pattern"],
                        "source_surface": "openai_responses",
                        "category": "chat",
                        "has_tools": False,
                        "stream": False,
                    },
                    "action": {
                        "type": "exact_cache_pattern",
                        "allow_tool_calls": False,
                        "safe_invalidation": False,
                        "streaming": False,
                    },
                }
            ]
        }
    else:
        target_policy_section = "anthropic_thinking_history_compaction.rules"
        local_policy_patch = {
            "anthropic_thinking_history_compaction": {
                "rules": [
                    {
                        "id": "managed-crunch-thinking-fixture",
                        "enabled": True,
                        "policy_source": "managed-recommended",
                        "candidate_id": "managed-crunch-thinking-fixture",
                        "conditions": {
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "text_bucket": "gte_128k_chars",
                            "model_pattern": "sonnet",
                            "has_tools": True,
                            "stream": True,
                        },
                        "action": {
                            "type": "compact_thinking_history_block",
                            "min_text_chars": 128000,
                            "min_block_chars": 2000,
                            "similarity_threshold": 0.95,
                            "preserve_tool_protocol": True,
                        },
                    }
                ]
            }
        }
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
            "target_local_policy_section": target_policy_section,
            "local_policy_patch": local_policy_patch,
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
    def setUp(self) -> None:
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        self.tmp = TemporaryDirectory()
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")

    def tearDown(self) -> None:
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        self.tmp.cleanup()

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
                self.assertIn("local_policy_patch", entry["local_policy_draft"])

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

    def test_apply_selected_staged_managed_activation_drafts_and_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            (config_dir / "cache_rules.yaml").write_text(
                "exact_cache:\n  enabled: true\npattern_rules: []\n",
                encoding="utf-8",
            )
            (config_dir / "crunch_rules.yaml").write_text(
                "enabled: true\nanthropic_thinking_history_compaction:\n  rules: []\n",
                encoding="utf-8",
            )

            staged = stage_managed_activation_bundle_sync(
                _bundle(),
                workspace=workspace,
                config_dir=config_dir,
            )
            self.assertTrue(staged["ok"], staged)

            for row in staged["staged"]:
                dry_run = apply_staged_managed_activation_bundle(
                    str(row["draft_id"]),
                    workspace=workspace,
                    config_dir=config_dir,
                    dry_run=True,
                )
                self.assertTrue(dry_run["ok"], dry_run)
                self.assertEqual(dry_run["status"], "dry-run")
                self.assertTrue(dry_run["files"][0]["diff"])
                self.assertFalse(dry_run["privacy"]["policy_files_written"])

                applied = apply_staged_managed_activation_bundle(
                    str(row["draft_id"]),
                    workspace=workspace,
                    config_dir=config_dir,
                    apply_id=f"apply-{row['local_action_family']}",
                )
                self.assertTrue(applied["ok"], applied)
                self.assertEqual(applied["status"], "applied")
                self.assertEqual(applied["changed_sections"], [row["local_action_family"]])
                self.assertTrue(applied["rollback_command"])
                self.assertFalse(applied["provider_calls_made"] if "provider_calls_made" in applied else False)
                self.assertFalse(applied["privacy"]["managed_server_calls_made"])

            cache_payload = yaml.safe_load((config_dir / "cache_rules.yaml").read_text(encoding="utf-8"))
            cache_rule = cache_payload["pattern_rules"][0]
            self.assertEqual(cache_rule["policy_source"], "managed-recommended")
            self.assertEqual(cache_rule["managed_activation_apply"]["action_id"], "local-activation-policy-bundle-action:cache:fixture")
            self.assertTrue(cache_rule["managed_activation_apply"]["rollback_ready"])

            crunch_payload = yaml.safe_load((config_dir / "crunch_rules.yaml").read_text(encoding="utf-8"))
            crunch_rule = crunch_payload["anthropic_thinking_history_compaction"]["rules"][0]
            self.assertEqual(crunch_rule["policy_source"], "managed-recommended")
            self.assertEqual(crunch_rule["managed_activation_apply"]["action_id"], "local-activation-policy-bundle-action:crunch:fixture")

            async def reload_state() -> dict[str, object]:
                policies: dict[str, object] = {}
                for section, filename in (("cache", "cache_rules.yaml"), ("crunch", "crunch_rules.yaml")):
                    path = config_dir / filename
                    digest = None
                    if path.exists():
                        import hashlib

                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    policies[section] = {
                        "file": {
                            "loaded": {"sha256": digest},
                            "current": {"sha256": digest},
                            "reload_required": False,
                        }
                    }
                return {"ok": True, "policies": policies}

            import asyncio

            for section in ("cache", "crunch"):
                rolled_back = asyncio.run(rollback_policy_apply(
                    f"apply-{section}",
                    config_dir=config_dir,
                    sections=[section],
                    reload_policy_state=reload_state,
                ))
                self.assertTrue(rolled_back["ok"], rolled_back)

            self.assertEqual(
                yaml.safe_load((config_dir / "cache_rules.yaml").read_text(encoding="utf-8"))["pattern_rules"],
                [],
            )
            self.assertEqual(
                yaml.safe_load((config_dir / "crunch_rules.yaml").read_text(encoding="utf-8"))["anthropic_thinking_history_compaction"]["rules"],
                [],
            )

    def test_apply_cli_writes_selected_managed_activation_draft(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            (config_dir / "cache_rules.yaml").write_text("pattern_rules: []\n", encoding="utf-8")

            staged = stage_managed_activation_bundle_sync(
                _bundle(),
                workspace=workspace,
                config_dir=config_dir,
            )
            cache_row = next(row for row in staged["staged"] if row["local_action_family"] == "cache")
            stdout = io.StringIO()

            code = cli.managed_activation_bundle_apply_cli(
                [
                    "--workspace",
                    str(workspace),
                    "--config-dir",
                    str(config_dir),
                    "--draft-id",
                    "cache-draft-fixture",
                    str(cache_row["draft_id"]),
                ],
                stdout=stdout,
            )

            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 0, result)
            self.assertTrue(result["ok"])
            payload = yaml.safe_load((config_dir / "cache_rules.yaml").read_text(encoding="utf-8"))
            self.assertEqual(payload["pattern_rules"][0]["draft_id"], "cache-draft-fixture")

    def test_apply_rejects_unsupported_target_rule_file(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            staged = stage_managed_activation_bundle_sync(
                _bundle(),
                workspace=workspace,
                config_dir=config_dir,
            )
            cache_row = next(row for row in staged["staged"] if row["local_action_family"] == "cache")
            bundle_path = Path(cache_row["stage"]["bundle_path"])
            bundle_payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
            bundle_payload["policies"]["cache"]["managed_activation_drafts"][0]["target_local_rule_file"] = "routing_rules.yaml"
            bundle_path.write_text(yaml.safe_dump(bundle_payload, sort_keys=False), encoding="utf-8")

            result = apply_staged_managed_activation_bundle(
                str(cache_row["draft_id"]),
                workspace=workspace,
                config_dir=config_dir,
            )

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["rejected"][0]["error"]["type"], "unsupported-target-rule-file")


if __name__ == "__main__":
    unittest.main()
