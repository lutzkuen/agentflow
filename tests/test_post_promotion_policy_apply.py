import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from tokenclaw.admin import reload_policy_modules
from tokenclaw.post_promotion_policy_apply import apply_post_promotion_policy_drafts


SENSITIVE_FIXTURES = (
    "raw prompt fixture must not leak",
    "raw response fixture must not leak",
    "raw provider body fixture must not leak",
    "raw tool payload fixture must not leak",
    "request-id-fixture-must-not-leak",
    "session-id-fixture-must-not-leak",
)


def _assert_metadata_only(testcase, payload):
    rendered = json.dumps(payload, sort_keys=True)
    for value in SENSITIVE_FIXTURES:
        testcase.assertNotIn(value, rendered)


def _gate(**overrides):
    gate = {
        "schema": "agentflow.post_promotion_policy_draft_impact_gate.v1",
        "status": "passed",
        "reason": "impact-gate-passed",
        "blocker_reasons": [],
        "affected_call_count": 20,
        "affected_row_count": 20,
        "current_canary_fraction": 0.1,
        "projected_canary_fraction": 0.2,
        "required_holdout_fraction": 0.1,
        "observed_holdout_fraction": 0.2,
        "holdout_coverage_present": True,
        "stale_evidence": False,
        "safety_stop_active": False,
        "preserved_previous_rule": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    gate.update(overrides)
    return gate


def _rollback_metadata():
    return {
        "schema": "agentflow.post_promotion_policy_draft_rollback_metadata.v1",
        "rollback_action_type": "disable_rule",
        "preserve_previous_rule_required": True,
        "preserve_operator_rule_history": True,
        "rollback_reason_codes": ["operator-requested"],
    }


def _draft(
    *,
    draft_id,
    action,
    family,
    candidate_id,
    policy_section,
    gate=None,
):
    return {
        "schema": "agentflow.post_promotion_policy_draft.v1",
        "status": "drafted",
        "draft_action": action,
        "draft_id": draft_id,
        "rule_id": draft_id,
        "target_candidate_id": candidate_id,
        "action_family": family,
        "target_local_rule_file": f"{family}_rules.yaml",
        "target_local_policy_section": policy_section,
        "source": "post-promotion-priority-delta-review",
        "proposed_policy_patch": {
            "schema": "agentflow.post_promotion_local_policy_patch.v1",
            "operation": "widen_existing_rule" if action == "widen-local-policy" else "rollback_existing_rule",
            "target_rule_selector": {
                "delta_id": candidate_id,
                "policy_section": policy_section,
            },
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "dry_run_impact_gate": gate or _gate(),
        "rollback_metadata": _rollback_metadata(),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }


def _report(*drafts):
    return {
        "schema": "agentflow.post_promotion_policy_draft_dry_run.v1",
        "ok": True,
        "status": "drafted",
        "generated_at": "2026-06-15T08:00:00+00:00",
        "drafts": list(drafts),
        "omitted": [],
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


class PostPromotionPolicyApplyTest(unittest.TestCase):
    def test_safe_widen_and_rollback_update_local_rule_files_with_rollback_metadata(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            workspace = Path(tmp) / "drafts"
            event_log = Path(tmp) / "policy_events.jsonl"
            config_dir.mkdir()
            routing_path = config_dir / "routing_rules.yaml"
            cache_path = config_dir / "cache_rules.yaml"
            crunch_path = config_dir / "crunch_rules.yaml"
            routing_path.write_text(
                yaml.safe_dump({
                    "rules": [
                        {
                            "id": "routing-widen-rule",
                            "candidate_id": "delta-widen",
                            "enabled": True,
                            "policy_source": "local-manual",
                            "conditions": {"model_pattern": "sonnet", "category": "tool-result"},
                            "action": {"route_to": "haiku", "reason": "fixture route"},
                            "canary": {"enabled": True, "canary_fraction": 0.1, "holdout_fraction": 0.2},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            cache_path.write_text(
                yaml.safe_dump({
                    "exact_cache": {"enabled": True, "cache_tool_calls": False},
                    "semantic_cache": {"enabled": False, "threshold": 0.95},
                    "pattern_rules": [
                        {
                            "id": "cache-rollback-rule",
                            "candidate_id": "delta-rollback",
                            "enabled": True,
                            "policy_source": "local-manual",
                            "conditions": {
                                "pattern_hashes": ["sha256:*"],
                                "source_surface": "anthropic_messages",
                                "category": "chat",
                                "has_tools": False,
                            },
                            "rollout": {"canary_enabled": True, "canary_fraction": 0.3, "holdout_fraction": 0.1},
                            "action": {"type": "exact_cache_pattern", "streaming": True, "allow_tool_calls": False},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            crunch_path.write_text("enabled: true\nthreshold_chars: 24000\n", encoding="utf-8")
            report = _report(
                _draft(
                    draft_id="routing-widen",
                    action="widen-local-policy",
                    family="routing",
                    candidate_id="delta-widen",
                    policy_section="routing.rules",
                    gate=_gate(current_canary_fraction=0.1, projected_canary_fraction=0.2),
                ),
                _draft(
                    draft_id="cache-rollback",
                    action="rollback-local-policy",
                    family="cache",
                    candidate_id="delta-rollback",
                    policy_section="cache.pattern_rules",
                    gate=_gate(holdout_coverage_present=True, projected_canary_fraction=0.0),
                ),
            )
            env = {
                "AGENTFLOW_ROUTING_RULES": str(routing_path),
                "AGENTFLOW_CACHE_RULES": str(cache_path),
                "AGENTFLOW_CRUNCH_RULES": str(crunch_path),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    result = asyncio.run(apply_post_promotion_policy_drafts(
                        report,
                        config_dir=config_dir,
                        workspace=workspace,
                        reload_policy_state=reload_policy_modules,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "applied")
            self.assertEqual(result["summary"]["applied_count"], 2)
            self.assertTrue(result["wrote_local_policy_files"])
            routing = yaml.safe_load(routing_path.read_text(encoding="utf-8"))
            routing_rule = routing["rules"][0]
            self.assertEqual(routing_rule["canary"]["canary_fraction"], 0.2)
            self.assertEqual(routing_rule["canary"]["holdout_fraction"], 0.2)
            self.assertEqual(routing_rule["post_promotion_policy_apply"]["draft_id"], "routing-widen")
            self.assertEqual(routing_rule["post_promotion_policy_apply"]["previous_rule"]["canary"]["canary_fraction"], 0.1)
            cache = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
            cache_rule = cache["pattern_rules"][0]
            self.assertFalse(cache_rule["enabled"])
            self.assertFalse(cache_rule["rollout"]["canary_enabled"])
            self.assertEqual(cache_rule["rollout"]["canary_fraction"], 0.0)
            self.assertEqual(cache_rule["post_promotion_policy_apply"]["rollback_metadata"]["rollback_action_type"], "disable_rule")
            backups = [
                Path(backup["path"])
                for outcome in result["outcomes"]
                for backup in (outcome.get("apply") or {}).get("backups", [])
            ]
            self.assertEqual(len(backups), 2)
            self.assertTrue(all(path.exists() for path in backups))
            _assert_metadata_only(self, result)

    def test_refuses_stale_missing_holdout_and_safety_stop_without_writes(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            workspace = Path(tmp) / "drafts"
            config_dir.mkdir()
            routing_path = config_dir / "routing_rules.yaml"
            routing_original = yaml.safe_dump({
                "rules": [
                    {
                        "id": "routing-widen-rule",
                        "candidate_id": "delta-widen",
                        "conditions": {"model_pattern": "sonnet", "category": "tool-result"},
                        "action": {"route_to": "haiku", "reason": "fixture route"},
                        "canary": {"enabled": True, "canary_fraction": 0.1, "holdout_fraction": 0.2},
                    }
                ],
            })
            routing_path.write_text(routing_original, encoding="utf-8")
            report = _report(
                _draft(
                    draft_id="stale",
                    action="widen-local-policy",
                    family="routing",
                    candidate_id="delta-widen",
                    policy_section="routing.rules",
                    gate=_gate(stale_evidence=True),
                ),
                _draft(
                    draft_id="missing-holdout",
                    action="widen-local-policy",
                    family="routing",
                    candidate_id="delta-widen",
                    policy_section="routing.rules",
                    gate=_gate(holdout_coverage_present=False),
                ),
                _draft(
                    draft_id="safety-stop",
                    action="widen-local-policy",
                    family="routing",
                    candidate_id="delta-widen",
                    policy_section="routing.rules",
                    gate=_gate(safety_stop_active=True),
                ),
            )
            with patch.dict(os.environ, {"AGENTFLOW_ROUTING_RULES": str(routing_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    result = asyncio.run(apply_post_promotion_policy_drafts(
                        report,
                        config_dir=config_dir,
                        workspace=workspace,
                        reload_policy_state=reload_policy_modules,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["summary"]["blocked_count"], 3)
            self.assertFalse(result["wrote_local_policy_files"])
            self.assertEqual(routing_path.read_text(encoding="utf-8"), routing_original)
            error_types = {outcome["error"]["type"] for outcome in result["outcomes"]}
            self.assertEqual(error_types, {"stale_evidence", "missing_holdout_coverage", "safety_stop_active"})
            _assert_metadata_only(self, result)

    def test_dry_run_stages_without_active_rule_file_write(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".agentflow"
            workspace = Path(tmp) / "drafts"
            config_dir.mkdir()
            routing_path = config_dir / "routing_rules.yaml"
            original = yaml.safe_dump({
                "rules": [
                    {
                        "id": "routing-widen-rule",
                        "candidate_id": "delta-widen",
                        "conditions": {"model_pattern": "sonnet", "category": "tool-result"},
                        "action": {"route_to": "haiku", "reason": "fixture route"},
                        "canary": {"enabled": True, "canary_fraction": 0.1, "holdout_fraction": 0.2},
                    }
                ],
            })
            routing_path.write_text(original, encoding="utf-8")
            report = _report(_draft(
                draft_id="routing-widen",
                action="widen-local-policy",
                family="routing",
                candidate_id="delta-widen",
                policy_section="routing.rules",
            ))
            with patch.dict(os.environ, {"AGENTFLOW_ROUTING_RULES": str(routing_path)}, clear=False):
                asyncio.run(reload_policy_modules())
                try:
                    result = asyncio.run(apply_post_promotion_policy_drafts(
                        report,
                        config_dir=config_dir,
                        workspace=workspace,
                        dry_run=True,
                        reload_policy_state=reload_policy_modules,
                    ))
                finally:
                    asyncio.run(reload_policy_modules())

            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(result["summary"]["dry_run_count"], 1)
            self.assertFalse(result["wrote_local_policy_files"])
            self.assertEqual(routing_path.read_text(encoding="utf-8"), original)
            self.assertTrue((workspace / "post-promotion-routing-widen" / "draft.json").exists())
            _assert_metadata_only(self, result)


if __name__ == "__main__":
    unittest.main()
