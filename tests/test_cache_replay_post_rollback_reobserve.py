import unittest

from tokenclaw.cache_replay_post_rollback_reobserve import (
    DECISION_KEEP_BLOCKED,
    DECISION_RESTAGE,
    DECISION_RETIRE,
    SCHEMA,
    STATE_FRESH_REPEAT_NO_HIT_PROOF,
    STATE_HIT_RECOVERY_PROOF,
    STATE_INVALIDATION_BLOCKER,
    STATE_NO_FRESH_TRAFFIC,
    STATE_RESTAGE_READY,
    build_cache_replay_post_rollback_reobserve_report,
)
from tokenclaw.managed_egress import managed_egress_violations


def _no_repeat_cohort():
    return {
        "cohort_label": "cache-replay-shape-alpha",
        "request_shape_fingerprint": "sha256:rawalphafingerprintvalue000000",
        "applied_count": 0,
        "holdout_count": 0,
        "miss_count": 0,
        "warmup_miss_count": 0,
        "exact_hit_count": 0,
        "observed_row_count": 0,
        "observation_age_hours": 96.0,
        "max_observation_age_hours": 72.0,
    }


def _invalidated_cohort():
    return {
        "cohort_label": "cache-replay-shape-beta",
        "request_shape_fingerprint": "sha256:rawbetafingerprintvalue0000000",
        "applied_count": 4,
        "holdout_count": 1,
        "exact_hit_count": 0,
        "observed_row_count": 5,
        "fresh_repeat_count": 4,
        "invalidation_blocked": True,
        "invalidation_reason": "dependency-changed",
        "observation_age_hours": 96.0,
        "max_observation_age_hours": 72.0,
    }


def _fresh_repeat_cohort():
    return {
        "cohort_label": "cache-replay-shape-gamma",
        "request_shape_fingerprint": "sha256:rawgammafingerprintvalue000000",
        "applied_count": 6,
        "holdout_count": 2,
        "exact_hit_count": 5,
        "observed_hits": 5,
        "observed_savings_usd": 0.42,
        "observed_row_count": 8,
        "fresh_repeat_count": 7,
        "observation_age_hours": 12.0,
        "max_observation_age_hours": 72.0,
    }


class CacheReplayPostRollbackReobserveTest(unittest.TestCase):
    def test_one_decision_per_cohort_with_acceptance_metric(self):
        rows = [_no_repeat_cohort(), _invalidated_cohort(), _fresh_repeat_cohort()]
        report = build_cache_replay_post_rollback_reobserve_report(rows)

        self.assertEqual(report["schema"], SCHEMA)
        cohorts = report["cohorts"]
        # Exactly one decision per cohort.
        self.assertEqual(len(cohorts), 3)
        self.assertEqual(report["summary"]["cohort_count"], 3)
        self.assertEqual(report["summary"]["decisions_per_cohort"], 1)

        by_label = {cohort["cohort_label"]: cohort for cohort in cohorts}

        retire = by_label["cache-replay-shape-alpha"]
        self.assertEqual(retire["successor_decision"], DECISION_RETIRE)
        self.assertEqual(retire["state"], STATE_NO_FRESH_TRAFFIC)
        self.assertTrue(retire["terminal_successor_state"])
        self.assertTrue(retire["stale_no_traffic_retirement"])

        blocked = by_label["cache-replay-shape-beta"]
        self.assertEqual(blocked["successor_decision"], DECISION_KEEP_BLOCKED)
        self.assertEqual(blocked["state"], STATE_INVALIDATION_BLOCKER)
        self.assertEqual(blocked["blocker_codes"], ["cache-replay-invalidation-blocker"])

        restage = by_label["cache-replay-shape-gamma"]
        self.assertEqual(restage["successor_decision"], DECISION_RESTAGE)
        self.assertEqual(restage["state"], STATE_HIT_RECOVERY_PROOF)

        # Decisions are restricted to the three allowed successor decisions.
        self.assertEqual(
            {cohort["successor_decision"] for cohort in cohorts},
            {DECISION_RETIRE, DECISION_KEEP_BLOCKED, DECISION_RESTAGE},
        )

    def test_no_cache_apply_actions_or_cache_entries_written(self):
        rows = [_no_repeat_cohort(), _invalidated_cohort(), _fresh_repeat_cohort()]
        report = build_cache_replay_post_rollback_reobserve_report(rows)

        self.assertEqual(report["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(report["summary"]["cache_entries_written"], 0)
        self.assertFalse(report["summary"]["emits_cache_apply_action"])
        self.assertFalse(report["summary"]["policy_files_written"])
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["managed_server_calls_made"])

        for cohort in report["cohorts"]:
            self.assertEqual(cohort["cache_apply_action_count"], 0)
            self.assertEqual(cohort["cache_entries_written"], 0)
            self.assertFalse(cohort["emits_cache_apply_action"])
            self.assertFalse(cohort["policy_files_written"])
            self.assertFalse(cohort["provider_calls_made"])
            self.assertFalse(cohort["managed_server_calls_made"])

    def test_next_research_plan_suppresses_predecessor_stale_rollback_issue(self):
        rows = [_no_repeat_cohort(), _invalidated_cohort(), _fresh_repeat_cohort()]
        report = build_cache_replay_post_rollback_reobserve_report(rows)

        plan = report["next_research_plan"]
        self.assertTrue(plan["suppresses_predecessor_stale_rollback_issue"])
        suppression = plan["duplicate_suppression"]
        self.assertTrue(suppression["active"])
        self.assertTrue(suppression["suppresses_generic_cache_replay_activation_issue"])
        self.assertTrue(suppression["suppresses_closed_stage_replay_predecessor_titles"])
        self.assertIn(
            "apply-cache-replay-rollback-before-reobserve",
            suppression["suppressed_predecessor_next_actions"],
        )
        self.assertIn(
            "Apply preview-verified cache rollback patches through the local activation executor",
            suppression["suppressed_predecessor_title_families"],
        )
        self.assertEqual(suppression["retired_cohort_count"], 1)

    def test_no_retirement_means_no_predecessor_suppression(self):
        # Only a fresh-repeat (restage) cohort: nothing stale to retire.
        report = build_cache_replay_post_rollback_reobserve_report([_fresh_repeat_cohort()])
        plan = report["next_research_plan"]
        self.assertFalse(plan["suppresses_predecessor_stale_rollback_issue"])
        self.assertFalse(plan["duplicate_suppression"]["active"])
        self.assertEqual(plan["duplicate_suppression"]["retired_cohort_count"], 0)

    def test_stable_fingerprints_are_deterministic_and_non_raw(self):
        first = build_cache_replay_post_rollback_reobserve_report([_no_repeat_cohort()])
        second = build_cache_replay_post_rollback_reobserve_report([_no_repeat_cohort()])
        fp_first = first["cohorts"][0]["cohort_fingerprint"]
        fp_second = second["cohorts"][0]["cohort_fingerprint"]
        self.assertEqual(fp_first, fp_second)
        self.assertTrue(fp_first.startswith("cache-replay-cohort:"))
        # The raw sha256 fingerprint never appears verbatim.
        self.assertNotIn("rawalphafingerprintvalue", fp_first)

    def test_fresh_repeat_states_distinguish_keep_blocked_and_restage_ready(self):
        # Fresh repeats, no hit proof, window still open -> keep-blocked.
        open_window = {
            "cohort_label": "cache-replay-shape-open",
            "request_shape_fingerprint": "sha256:openshapefingerprint000000000",
            "applied_count": 3,
            "exact_hit_count": 0,
            "observed_row_count": 3,
            "fresh_repeat_count": 3,
            "observation_age_hours": 10.0,
            "max_observation_age_hours": 72.0,
        }
        # Mature fresh repeats past the window, still no hits -> restage-ready.
        mature = {
            "cohort_label": "cache-replay-shape-mature",
            "request_shape_fingerprint": "sha256:matureshapefingerprint0000000",
            "applied_count": 5,
            "exact_hit_count": 0,
            "observed_row_count": 5,
            "fresh_repeat_count": 5,
            "observation_age_hours": 96.0,
            "max_observation_age_hours": 72.0,
        }
        report = build_cache_replay_post_rollback_reobserve_report([open_window, mature])
        by_label = {cohort["cohort_label"]: cohort for cohort in report["cohorts"]}

        self.assertEqual(by_label["cache-replay-shape-open"]["state"], STATE_FRESH_REPEAT_NO_HIT_PROOF)
        self.assertEqual(
            by_label["cache-replay-shape-open"]["successor_decision"], DECISION_KEEP_BLOCKED
        )
        self.assertEqual(by_label["cache-replay-shape-mature"]["state"], STATE_RESTAGE_READY)
        self.assertEqual(
            by_label["cache-replay-shape-mature"]["successor_decision"], DECISION_RESTAGE
        )

    def test_duplicate_cohort_fingerprints_collapse_to_one_decision(self):
        rows = [_no_repeat_cohort(), _no_repeat_cohort()]
        report = build_cache_replay_post_rollback_reobserve_report(rows)
        self.assertEqual(len(report["cohorts"]), 1)
        self.assertEqual(report["summary"]["cohort_count"], 1)

    def test_report_has_no_managed_egress_violations(self):
        rows = [_no_repeat_cohort(), _invalidated_cohort(), _fresh_repeat_cohort()]
        report = build_cache_replay_post_rollback_reobserve_report(rows)
        self.assertEqual(managed_egress_violations(report), [])
        self.assertTrue(report["privacy"]["egress_safe"])


if __name__ == "__main__":
    unittest.main()
