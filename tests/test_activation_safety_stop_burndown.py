import io
import json
from pathlib import Path
import tempfile
import unittest

from agentflow_proxy import cli
from agentflow_proxy.activation_lifecycle_feedback import (
    LIFECYCLE_SOURCE_SURFACE,
    activation_safety_stop_burndown_report,
    build_activation_safety_stop_burndown,
    build_activation_staged_lifecycle_feedback,
)
from agentflow_proxy.store import Store, stable_json, utc_now


class ActivationSafetyStopBurndownTests(unittest.TestCase):
    def test_lifecycle_safety_stop_groups_have_specific_next_action(self):
        result = {
            "schema": "fixture.activation.apply.v1",
            "ok": False,
            "summary": {"projected_savings_usd": 0.04},
            "actions": [
                {
                    "action_family": "routing",
                    "policy_section": "routing",
                    "status": "safety_stopped",
                    "target_candidate_id": "raw-candidate-secret",
                    "target_rule_id": "raw-rule-secret",
                    "projected_savings_usd": 0.04,
                    "reason_codes": ["local-canary-safety-stop", "error-rate-regression"],
                }
            ],
        }
        payload = build_activation_staged_lifecycle_feedback(
            result,
            event_phase="apply",
            command="activation-apply-fixture",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["family_events"][0]["cohort"], "safety_stopped")

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="activation-safety-stop",
                    created_at=now,
                    updated_at=now,
                    source_surface=LIFECYCLE_SOURCE_SURFACE,
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                report = activation_safety_stop_burndown_report(store, limit=50)
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        group = report["groups"][0]
        self.assertEqual(group["action_family"], "routing")
        self.assertEqual(group["blocker_code"], "error-rate-regression")
        self.assertIn("rollback_proof", group["needed_resolution"])
        self.assertEqual(group["next_action"], "record-routing-rollback-proof-before-reactivation")
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-candidate-secret", rendered)
        self.assertNotIn("raw-rule-secret", rendered)

    def test_repeated_research_diagnostic_resolves_to_action_without_raw_examples(self):
        plan = {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {
                "repeated_diagnostics": [
                    {
                        "reason": "safety-stop",
                        "diagnostic_class": "safety-stop",
                        "count": 8,
                        "example": "routing blocker=safety-stop request_id=req-secret path=/tmp/raw.py session_id=session-secret",
                    }
                ]
            },
        }

        report = build_activation_safety_stop_burndown(research_plan=plan)

        self.assertEqual(report["status"], "ranked")
        self.assertEqual(report["summary"]["top_next_action"], "review-activation-feedback-safety-stop-and-record-keep-blocked-reason")
        self.assertEqual(report["groups"][0]["repeated_noop_status"], "repeated")
        self.assertIn("human_review", report["groups"][0]["needed_resolution"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("req-secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("/tmp/raw.py", rendered)

    def test_cli_reads_plan_json_and_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "agentflow.sqlite3"
            store = Store(str(db_path))
            store.conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema": "agentflow.orchestrator_research_plan.v1",
                        "evidence": {
                            "repeated_diagnostics": [
                                {"reason": "safety-stop", "diagnostic_class": "safety-stop", "count": 3}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            code = cli.activation_safety_stop_burndown_cli(
                ["--db", str(db_path), "--plan-json", str(plan_path), "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["summary"]["safety_stop_count"], 3)
        self.assertEqual(report["summary"]["top_next_action"], "review-activation-feedback-safety-stop-and-record-keep-blocked-reason")


if __name__ == "__main__":
    unittest.main()
