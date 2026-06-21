from __future__ import annotations

import copy
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_managed_activation_bundles import _bundle  # noqa: E402

from tokenclaw.cli_commands import optimization_reports as optimization_reports_cli
from tokenclaw.local_activation_outcomes import build_local_activation_outcome_summary
from tokenclaw.managed_activation_apply import apply_staged_managed_activation_bundle
from tokenclaw.managed_activation_bundle_apply_outcomes import (
    build_managed_activation_bundle_apply_outcomes,
)
from tokenclaw.managed_activation_bundles import stage_managed_activation_bundle_sync
from tokenclaw.policy_events import recent_policy_events
from tokenclaw.policy_workbench import rollback_policy_apply
from tokenclaw.store import Store


async def _reload_state(config_dir: Path) -> dict[str, object]:
    import asyncio
    import hashlib

    policies: dict[str, object] = {}
    for section, filename in (("cache", "cache_rules.yaml"), ("crunch", "crunch_rules.yaml")):
        path = config_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        policies[section] = {
            "file": {
                "loaded": {"sha256": digest},
                "current": {"sha256": digest},
                "reload_required": False,
            }
        }
    return {"ok": True, "policies": policies}


class ManagedActivationBundleApplyOutcomesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_event_log = os.environ.get("TOKENCLAW_POLICY_EVENTS_LOG")
        self.tmp = TemporaryDirectory()
        os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")

    def tearDown(self) -> None:
        if self.old_event_log is None:
            os.environ.pop("TOKENCLAW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = self.old_event_log
        self.tmp.cleanup()

    def test_apply_skip_fail_and_rollback_produce_metadata_only_outcome_rows(self) -> None:
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

            staged = stage_managed_activation_bundle_sync(_bundle(), workspace=workspace, config_dir=config_dir)
            self.assertTrue(staged["ok"], staged)

            cache_row = next(row for row in staged["staged"] if row["local_action_family"] == "cache")
            crunch_row = next(row for row in staged["staged"] if row["local_action_family"] == "crunch")

            # Add a second cache managed-activation entry to the staged cache bundle so a
            # single apply call can both apply one entry and skip the other (not-selected).
            bundle_path = Path(cache_row["stage"]["bundle_path"])
            bundle_payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
            cache_drafts = bundle_payload["policies"]["cache"]["managed_activation_drafts"]
            original_action_id = cache_drafts[0]["action_id"]
            second_entry = copy.deepcopy(cache_drafts[0])
            second_entry["action_id"] = "local-activation-policy-bundle-action:cache:second"
            second_entry["draft_id"] = "cache-draft-fixture-second"
            second_entry["recommendation_id"] = "cache-recommendation-fixture-second"
            second_entry["local_policy_draft"]["local_policy_patch"]["pattern_rules"][0]["id"] = "managed-cache-replay-fixture-second"
            second_entry["local_policy_draft"]["local_policy_patch"]["pattern_rules"][0]["candidate_id"] = "managed-cache-replay-fixture-second"
            cache_drafts.append(second_entry)
            bundle_path.write_text(yaml.safe_dump(bundle_payload, sort_keys=False), encoding="utf-8")

            applied_call = apply_staged_managed_activation_bundle(
                str(cache_row["draft_id"]),
                workspace=workspace,
                config_dir=config_dir,
                apply_id="apply-cache-mixed",
                selectors=[original_action_id],
            )
            self.assertTrue(applied_call["ok"], applied_call)
            self.assertEqual(len(applied_call["applied"]), 1)
            self.assertEqual(len(applied_call["skipped"]), 1)

            # Break only the second entry's target rule file, then apply selecting only it:
            # this rejects with an "unsupported-target-rule-file" failure and skips the first.
            bundle_payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
            cache_drafts = bundle_payload["policies"]["cache"]["managed_activation_drafts"]
            broken_entry = next(entry for entry in cache_drafts if entry["action_id"] == second_entry["action_id"])
            broken_entry["target_local_rule_file"] = "routing_rules.yaml"
            bundle_path.write_text(yaml.safe_dump(bundle_payload, sort_keys=False), encoding="utf-8")

            failed_call = apply_staged_managed_activation_bundle(
                str(cache_row["draft_id"]),
                workspace=workspace,
                config_dir=config_dir,
                apply_id="apply-cache-failed",
                selectors=[second_entry["action_id"]],
            )
            self.assertFalse(failed_call["ok"], failed_call)
            self.assertEqual(failed_call["rejected"][0]["error"]["type"], "unsupported-target-rule-file")

            crunch_applied = apply_staged_managed_activation_bundle(
                str(crunch_row["draft_id"]),
                workspace=workspace,
                config_dir=config_dir,
                apply_id="apply-crunch",
            )
            self.assertTrue(crunch_applied["ok"], crunch_applied)

            import asyncio

            rolled_back = asyncio.run(rollback_policy_apply(
                "apply-cache-mixed",
                config_dir=config_dir,
                sections=["cache"],
                reload_policy_state=lambda: _reload_state(config_dir),
            ))
            self.assertTrue(rolled_back["ok"], rolled_back)

            events = recent_policy_events(limit=50)["events"]
            self.assertGreaterEqual(len(events), 4)

            outcomes = build_managed_activation_bundle_apply_outcomes(events)
            self.assertEqual(outcomes["schema"], "tokenclaw.managed_activation_bundle_apply_outcomes.v1")
            self.assertEqual(outcomes["egress_guard"]["status"], "passed")
            self.assertTrue(outcomes["privacy"]["feature_only"])
            self.assertTrue(outcomes["privacy"]["metadata_only"])
            self.assertTrue(outcomes["privacy"]["aggregate_only"])
            self.assertFalse(outcomes["privacy"]["raw_prompts_included"])
            self.assertFalse(outcomes["privacy"]["cache_keys_included"])
            self.assertFalse(outcomes["privacy"]["session_ids_included"])
            self.assertFalse(outcomes["privacy"]["tenant_ids_included"])

            by_key = {(row["local_action_family"], row["apply_status"]): row for row in outcomes["outcomes"]}
            self.assertEqual(set(by_key), {
                ("cache", "rolled-back"),
                ("cache", "skipped"),
                ("cache", "failed"),
                ("crunch", "applied"),
            })
            self.assertNotIn(("cache", "applied"), by_key)

            self.assertEqual(by_key[("cache", "rolled-back")]["rolled_back_count"], 1)
            self.assertEqual(by_key[("cache", "rolled-back")]["draft_id"], "cache-draft-fixture")
            self.assertEqual(by_key[("cache", "skipped")]["skipped_count"], 2)
            self.assertEqual(by_key[("cache", "failed")]["failed_count"], 1)
            self.assertEqual(by_key[("cache", "failed")]["blocker_codes"][0]["code"], "unsupported-target-rule-file")
            self.assertEqual(by_key[("crunch", "applied")]["applied_count"], 1)
            self.assertEqual(by_key[("crunch", "applied")]["draft_id"], "crunch-draft-fixture")

            # Idempotent / duplicate-safe: rebuilding from the same events yields the same rows.
            replay = build_managed_activation_bundle_apply_outcomes(events)
            self.assertEqual(outcomes["outcomes"], replay["outcomes"])
            self.assertEqual(outcomes["summary"], replay["summary"])

            rendered = json.dumps(outcomes, sort_keys=True)
            for forbidden in (str(Path(tmp).resolve()), str(workspace), str(config_dir)):
                self.assertNotIn(forbidden, rendered)

            with TemporaryDirectory() as db_tmp:
                store = Store(str(Path(db_tmp) / "tokenclaw.sqlite3"))
                try:
                    summary = build_local_activation_outcome_summary(
                        store,
                        limit=10,
                        config_dir=db_tmp,
                        managed_activation_bundle_apply_events=events,
                    )
                finally:
                    store.conn.close()
            self.assertEqual(summary["egress_guard"]["status"], "passed")
            self.assertIn("managed_activation_bundle_apply_outcomes", summary)
            self.assertEqual(
                summary["managed_activation_bundle_apply_outcomes"]["summary"]["outcome_row_count"],
                4,
            )

            stdout = io.StringIO()
            code = optimization_reports_cli.managed_activation_bundle_apply_outcomes_cli(
                ["--limit", "50"], stdout=stdout
            )
            self.assertEqual(code, 0)
            cli_payload = json.loads(stdout.getvalue())
            self.assertEqual(cli_payload["schema"], "tokenclaw.managed_activation_bundle_apply_outcomes.v1")
            self.assertEqual(cli_payload["summary"]["outcome_row_count"], 4)


if __name__ == "__main__":
    unittest.main()
