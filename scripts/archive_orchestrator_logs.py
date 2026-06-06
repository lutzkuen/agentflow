#!/usr/bin/env python3
"""Archive AgentFlow orchestrator logs without calling an LLM.

Quota-only logs are disposable after a short retention window. Logs that show
real orchestration work are kept, but moved into YYYY/MM/DD folders so the top
level log directory stays small.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LOG_NAME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
    r"\d{2}-\d{2}-\d{2}\.log$"
)
QUOTA_MARKERS = (
    "CLAUDE_RATE_LIMITED",
    "WORKER_RATE_LIMITED",
    "Recent Claude rate-limit cooldown is active",
    "Recent Codex rate-limit cooldown is active",
    "Recent worker rate-limit cooldown is active",
    "temporarily rate-limiting requests",
    "temporarily limiting requests",
    "account's rate limit",
    "account.s rate limit",
)
WORK_MARKERS = (
    "Running Claude orchestrator",
    "Running Codex orchestrator",
    "Starting a fresh Claude operator session",
    "Starting a fresh Codex operator session",
    "Creating isolated run worktree",
    "Run complete.",
    "CODEX_REQUIRED",
    "Codex recovery summary",
    "Merged `agent/",
)


@dataclass(frozen=True)
class ArchiveResult:
    archived: int = 0
    deleted_quota: int = 0
    kept_quota: int = 0
    kept_recent_work: int = 0
    kept_unfinished: int = 0
    skipped: int = 0

    def __add__(self, other: "ArchiveResult") -> "ArchiveResult":
        return ArchiveResult(
            archived=self.archived + other.archived,
            deleted_quota=self.deleted_quota + other.deleted_quota,
            kept_quota=self.kept_quota + other.kept_quota,
            kept_recent_work=self.kept_recent_work + other.kept_recent_work,
            kept_unfinished=self.kept_unfinished + other.kept_unfinished,
            skipped=self.skipped + other.skipped,
        )


def parse_log_date(path: Path) -> tuple[str, str, str] | None:
    match = LOG_NAME_RE.match(path.name)
    if not match:
        return None
    return match.group("year"), match.group("month"), match.group("day")


def read_log_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_quota_only_log(text: str) -> bool:
    has_quota_marker = any(marker in text for marker in QUOTA_MARKERS)
    has_work_marker = any(marker in text for marker in WORK_MARKERS)
    return has_quota_marker and not has_work_marker


def is_finished_log(text: str) -> bool:
    return "\nFinished:" in text or text.startswith("Finished:")


def age_hours(path: Path, now: datetime) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (now - mtime).total_seconds() / 3600


def archive_destination(path: Path, date_parts: tuple[str, str, str]) -> Path:
    year, month, day = date_parts
    return path.parent / year / month / day / path.name


def unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    for index in range(1, 1000):
        candidate = destination.with_name(f"{destination.stem}.{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free archive destination for {destination}")


def move_log(path: Path, destination: Path, dry_run: bool) -> Path:
    if destination.exists() and filecmp.cmp(path, destination, shallow=False):
        if not dry_run:
            path.unlink()
        return destination

    final_destination = unique_destination(destination)
    if not dry_run:
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(final_destination))
    return final_destination


def update_latest_symlink(log_dir: Path, moves: dict[Path, Path], dry_run: bool) -> None:
    latest = log_dir / "latest.log"
    if not latest.is_symlink():
        return

    try:
        target = latest.resolve(strict=False)
    except OSError:
        return

    new_target = moves.get(target)
    if new_target is None:
        return

    if not dry_run:
        latest.unlink()
        latest.symlink_to(new_target)


def process_log(
    path: Path,
    now: datetime,
    quota_retention_hours: float,
    work_archive_min_age_hours: float,
    dry_run: bool,
) -> tuple[ArchiveResult, Path | None]:
    date_parts = parse_log_date(path)
    if date_parts is None or path.is_symlink() or not path.is_file():
        return ArchiveResult(skipped=1), None

    text = read_log_text(path)
    if not is_finished_log(text):
        return ArchiveResult(kept_unfinished=1), None

    current_age_hours = age_hours(path, now)
    if is_quota_only_log(text):
        if current_age_hours > quota_retention_hours:
            if not dry_run:
                path.unlink()
            return ArchiveResult(deleted_quota=1), None
        return ArchiveResult(kept_quota=1), None

    if current_age_hours < work_archive_min_age_hours:
        return ArchiveResult(kept_recent_work=1), None

    destination = archive_destination(path, date_parts)
    if destination == path:
        return ArchiveResult(skipped=1), None

    final_destination = move_log(path, destination, dry_run)
    return ArchiveResult(archived=1), final_destination


def archive_orchestrator_logs(
    log_dir: Path,
    quota_retention_hours: float = 24,
    work_archive_min_age_hours: float = 2,
    now: datetime | None = None,
    dry_run: bool = False,
) -> ArchiveResult:
    now = now or datetime.now(timezone.utc)
    log_dir = log_dir.resolve()
    result = ArchiveResult()
    moves: dict[Path, Path] = {}

    for path in sorted(log_dir.iterdir()):
        item_result, destination = process_log(
            path,
            now,
            quota_retention_hours,
            work_archive_min_age_hours,
            dry_run,
        )
        result += item_result
        if destination is not None:
            moves[path.resolve(strict=False)] = destination.resolve(strict=False)

    update_latest_symlink(log_dir, moves, dry_run)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive AgentFlow orchestrator logs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("AGENTFLOW_ORCHESTRATOR_LOG_DIR", "logs/orchestrator")),
        help="Directory containing orchestrator logs.",
    )
    parser.add_argument(
        "--quota-retention-hours",
        type=float,
        default=float(os.environ.get("AGENTFLOW_QUOTA_LOG_RETENTION_HOURS", "24")),
        help="Hours to keep quota-only orchestrator logs.",
    )
    parser.add_argument(
        "--work-archive-min-age-hours",
        type=float,
        default=float(os.environ.get("AGENTFLOW_WORK_LOG_ARCHIVE_MIN_AGE_HOURS", "2")),
        help="Minimum age before moving completed work logs into YYYY/MM/DD.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without changing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = archive_orchestrator_logs(
        log_dir=args.log_dir,
        quota_retention_hours=args.quota_retention_hours,
        work_archive_min_age_hours=args.work_archive_min_age_hours,
        dry_run=args.dry_run,
    )
    print(
        "orchestrator log archive: "
        f"archived={result.archived} "
        f"deleted_quota={result.deleted_quota} "
        f"kept_quota={result.kept_quota} "
        f"kept_recent_work={result.kept_recent_work} "
        f"kept_unfinished={result.kept_unfinished} "
        f"skipped={result.skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
