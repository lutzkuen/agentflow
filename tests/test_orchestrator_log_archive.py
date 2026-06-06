from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.archive_orchestrator_logs import archive_orchestrator_logs, is_quota_only_log


HEADER = """## AgentFlow Cron Run
Started: 2026-06-06T05:17:01+02:00
Host: test-host
Repo: /tmp/agentflow
Run ID: 2026-06-06_05-17-01

"""
FOOTER = """
Finished: 2026-06-06T05:31:14+02:00
Exit: 0
"""


class OrchestratorLogArchiveTest(unittest.TestCase):
    def test_quota_log_detection_excludes_partial_work(self):
        quota_text = HEADER + "CLAUDE_RATE_LIMITED: cooldown active after recent upstream rate limit.\n" + FOOTER
        partial_work_text = quota_text + "CODEX_REQUIRED: Claude was rate-limited after partial work.\n"

        self.assertTrue(is_quota_only_log(quota_text))
        self.assertFalse(is_quota_only_log(partial_work_text))

    def test_deletes_old_quota_only_logs(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            old_quota = log_dir / "2026-06-05_01-17-01.log"
            old_quota.write_text(
                HEADER + "CLAUDE_RATE_LIMITED: cooldown active after recent upstream rate limit.\n" + FOOTER,
                encoding="utf-8",
            )

            result = archive_orchestrator_logs(
                log_dir,
                quota_retention_hours=24,
                now=datetime.now(timezone.utc) + timedelta(hours=25),
            )

            self.assertFalse(old_quota.exists())
            self.assertEqual(result.deleted_quota, 1)

    def test_keeps_recent_quota_only_logs(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            recent_quota = log_dir / "2026-06-06_01-17-01.log"
            recent_quota.write_text(
                HEADER + "CLAUDE_RATE_LIMITED: cooldown active after recent upstream rate limit.\n" + FOOTER,
                encoding="utf-8",
            )

            result = archive_orchestrator_logs(log_dir, quota_retention_hours=24, now=datetime.now(timezone.utc))

            self.assertTrue(recent_quota.exists())
            self.assertEqual(result.kept_quota, 1)

    def test_archives_work_logs_by_filename_date_and_updates_latest(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            work_log = log_dir / "2026-06-06_05-17-01.log"
            work_log.write_text(HEADER + "[2026-06-06 05:17:05] Running Claude orchestrator\n" + FOOTER, encoding="utf-8")
            latest = log_dir / "latest.log"
            latest.symlink_to(work_log)

            result = archive_orchestrator_logs(log_dir, work_archive_min_age_hours=0)

            archived = log_dir / "2026" / "06" / "06" / work_log.name
            self.assertFalse(work_log.exists())
            self.assertTrue(archived.exists())
            self.assertEqual(latest.resolve(strict=False), archived.resolve(strict=False))
            self.assertEqual(result.archived, 1)

    def test_keeps_unfinished_work_log_in_place(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            work_log = log_dir / "2026-06-06_06-17-01.log"
            work_log.write_text(HEADER + "[2026-06-06 06:17:07] Running Claude orchestrator\n", encoding="utf-8")

            result = archive_orchestrator_logs(log_dir, work_archive_min_age_hours=0)

            self.assertTrue(work_log.exists())
            self.assertEqual(result.kept_unfinished, 1)

    def test_keeps_recent_finished_work_log_in_place(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            work_log = log_dir / "2026-06-06_06-17-01.log"
            work_log.write_text(HEADER + "[2026-06-06 06:17:07] Running Claude orchestrator\n" + FOOTER, encoding="utf-8")

            result = archive_orchestrator_logs(
                log_dir,
                work_archive_min_age_hours=2,
                now=datetime.now(timezone.utc),
            )

            self.assertTrue(work_log.exists())
            self.assertEqual(result.kept_recent_work, 1)


if __name__ == "__main__":
    unittest.main()
