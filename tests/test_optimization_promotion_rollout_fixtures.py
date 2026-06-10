import asyncio
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.optimization_promotion_actions import build_optimization_promotion_actions
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
            self.assertEqual(dry_run["summary"]["planned_action_count"], 1)
            self.assertEqual(dry_run["summary"]["skipped_action_count"], 1)
            skipped = next(row for row in dry_run["actions"] if row["status"] == "skipped")
            self.assertEqual(skipped["policy_section"], "cache")
            self.assertEqual(skipped["reason"], "not-requested")
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())

        unsafe_bundle = copy.deepcopy(bundle)
        unsafe_bundle["actions"][0]["raw_request"] = {"prompt": "raw fixture prompt secret"}
        with tempfile.TemporaryDirectory() as tmp:
            rejected = apply_optimization_promotion_canaries(unsafe_bundle, config_dir=tmp, dry_run=True)
        self.assertFalse(rejected["ok"])
        self.assertIn("raw or local-identifier promotion rollout payloads are not accepted", {error["message"] for error in rejected["errors"]})

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
