import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.orchestrator_research import build_research_plan


NOW = datetime(2026, 6, 11, 8, 40, tzinfo=timezone.utc)


def issue(number, title, labels, *, repo="lutzkuen/agentflow", author="lutzkuen", state="OPEN", updated="2026-06-11T08:00:00Z"):
    return {
        "repo": repo,
        "number": number,
        "title": title,
        "state": state,
        "url": f"https://github.com/{repo}/issues/{number}",
        "author": {"login": author},
        "labels": [{"name": name} for name in labels],
        "updatedAt": updated,
    }


class OrchestratorResearchPlanTests(unittest.TestCase):
    def test_no_ready_issues_enters_research_and_creates_actionable_issue(self):
        plan = build_research_plan(
            issues=[
                issue(10, "Blocked old milestone", ["status:blocked", "priority:p1"], updated="2026-05-01T08:00:00Z"),
                issue(11, "External idea", ["status:ready", "priority:p1"], author="external"),
            ],
            stats={"calls": 5266, "cache_hit_rate": 0.0, "today_cost_usd": 12.34},
            threshold=3,
            now=NOW,
        )

        self.assertTrue(plan["research_trigger"]["should_run"])
        self.assertEqual(plan["research_trigger"]["reason"], "ready-actionable-count-below-threshold")
        self.assertEqual(plan["research_trigger"]["actionable_ready_count"], 0)
        created = plan["backlog_changes"]["create_issues"]
        self.assertGreaterEqual(len(created), 1)
        self.assertIn("Acceptance Criteria", created[0]["body"])
        self.assertIn("Implementation Approach", created[0]["body"])
        self.assertIn("status:ready", created[0]["labels"])

    def test_enough_ready_issues_is_noop(self):
        plan = build_research_plan(
            issues=[
                issue(1, "Ready one", ["status:ready", "priority:p1"]),
                issue(2, "Ready two", ["status:ready", "priority:p2"]),
                issue(3, "Ready three", ["status:ready", "priority:p3"]),
                issue(4, "Blocked", ["status:blocked", "priority:p1"], updated="2026-05-01T08:00:00Z"),
            ],
            threshold=3,
            now=NOW,
        )

        self.assertFalse(plan["research_trigger"]["should_run"])
        self.assertEqual(plan["backlog_changes"]["create_issues"], [])
        self.assertEqual(plan["backlog_changes"]["comment_issues"], [])
        self.assertIn("should not run", plan["run_log_summary"])

    def test_stale_blocked_issues_get_current_evidence_comment(self):
        plan = build_research_plan(
            issues=[
                issue(
                    220,
                    "Milestone: Local workflow phase memory",
                    ["status:blocked", "priority:p1", "core-feature"],
                    updated="2026-05-20T00:00:00Z",
                )
            ],
            stats={"calls": 20, "today_errors": 2},
            threshold=2,
            stale_days=14,
            now=NOW,
        )

        comments = plan["backlog_changes"]["comment_issues"]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]["number"], 220)
        self.assertIn("Blocked issue has been stale", comments[0]["body"])
        self.assertIn("Acceptance Criteria", comments[0]["body"])

    def test_repeated_skip_diagnostics_become_a_targeted_proposal(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "rollout skipped skip_reason=missing-dependency-evidence request_id=req-secret-12345",
                        "candidate omitted reason=missing-dependency-evidence session_id=session-secret-67890",
                        "quality gate blocked blocker=need-more-samples",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        self.assertEqual(diagnostics[0]["reason"], "missing-dependency-evidence")
        self.assertGreaterEqual(diagnostics[0]["count"], 2)
        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertTrue(any("missing dependency evidence" in title for title in created_titles))
        rendered = json.dumps(plan)
        self.assertNotIn("req-secret-12345", rendered)
        self.assertNotIn("session-secret-67890", rendered)

    def test_privacy_redacts_raw_fields_paths_and_ids(self):
        plan = build_research_plan(
            issues=[
                {
                    "repo": "lutzkuen/agentflow",
                    "number": 9,
                    "title": "Raw prompt must not leak",
                    "state": "OPEN",
                    "author": {"login": "lutzkuen"},
                    "labels": [{"name": "status:blocked"}],
                    "updatedAt": "2026-05-01T00:00:00Z",
                    "request_json": {"messages": [{"content": "private prompt text"}]},
                    "session_id": "session-raw-secret",
                }
            ],
            stats={
                "calls": 1,
                "request_json": {"messages": [{"content": "private stats prompt"}]},
                "routing": [{"requested_model": "gpt-5", "path": "/home/lutz/private/project/file.py"}],
            },
            log_sources=[
                "skip_reason=privacy-blocked request_id=req-raw-secret /home/lutz/private/project/file.py sk-testsecret123456"
            ],
            threshold=2,
            now=NOW,
        )

        rendered = json.dumps(plan)
        self.assertNotIn("private prompt text", rendered)
        self.assertNotIn("private stats prompt", rendered)
        self.assertNotIn("/home/lutz/private/project/file.py", rendered)
        self.assertNotIn("req-raw-secret", rendered)
        self.assertNotIn("session-raw-secret", rendered)
        self.assertNotIn("sk-testsecret123456", rendered)
        self.assertIn("[REDACTED", rendered)
        self.assertFalse(plan["privacy"]["raw_prompts_included"])
        self.assertFalse(plan["privacy"]["absolute_paths_included"])


class OrchestratorResearchCliTests(unittest.TestCase):
    def test_cli_reads_json_files_and_emits_plan(self):
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp) / "issues.json"
            stats_path = Path(tmp) / "stats.json"
            issues_path.write_text(json.dumps([issue(1, "Blocked", ["status:blocked"], updated="2026-05-01T00:00:00Z")]), encoding="utf-8")
            stats_path.write_text(json.dumps({"calls": 5, "cache_hit_rate": 0.0}), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            code = cli.orchestrator_research_cli(
                ["--issues-json", str(issues_path), "--stats-json", str(stats_path), "--threshold", "2"],
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.orchestrator_research_plan.v1")
        self.assertTrue(payload["research_trigger"]["should_run"])


if __name__ == "__main__":
    unittest.main()
