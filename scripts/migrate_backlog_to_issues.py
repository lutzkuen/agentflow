#!/usr/bin/env python3
"""Create GitHub Issues from non-DONE BACKLOG.md items."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ITEM_RE = re.compile(r"^- \[(?P<status>[A-Z-]+)\] (?P<title>.+?)\s*$")
SECTION_RE = re.compile(r"^## (?P<priority>P[0-9])\s+—\s+(?P<title>.+?)\s*$")


@dataclass
class BacklogItem:
    status: str
    title: str
    priority: str
    section: str
    body_lines: list[str]


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def parse_backlog(path: Path) -> list[BacklogItem]:
    items: list[BacklogItem] = []
    priority = "PX"
    section = "Uncategorized"
    current: BacklogItem | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
      section_match = SECTION_RE.match(line)
      if section_match:
          priority = section_match.group("priority")
          section = section_match.group("title")

      item_match = ITEM_RE.match(line)
      if item_match:
          if current is not None:
              items.append(current)
          current = BacklogItem(
              status=item_match.group("status"),
              title=item_match.group("title").strip(),
              priority=priority,
              section=section,
              body_lines=[],
          )
          continue

      if current is not None:
          current.body_lines.append(line)

    if current is not None:
        items.append(current)
    return items


def ensure_label(repo: str, name: str, color: str, description: str) -> None:
    result = run(
        [
            "gh",
            "label",
            "create",
            name,
            "--repo",
            repo,
            "--color",
            color,
            "--description",
            description,
        ],
        check=False,
    )
    if result.returncode != 0 and "already exists" not in result.stderr.lower():
        print(result.stderr.strip(), file=sys.stderr)


def existing_issue_titles(repo: str) -> set[str]:
    result = run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "title",
        ]
    )
    return {row["title"] for row in json.loads(result.stdout)}


def issue_body(item: BacklogItem) -> str:
    details = "\n".join(line.rstrip() for line in item.body_lines).strip()
    return (
        f"Imported from `BACKLOG.md`.\n\n"
        f"- Status: `{item.status}`\n"
        f"- Priority: `{item.priority}`\n"
        f"- Section: `{item.section}`\n\n"
        f"## Original Notes\n\n"
        f"{details or '(no additional notes)'}\n"
    )


def create_issue(repo: str, item: BacklogItem, dry_run: bool) -> None:
    labels = ["backlog", f"status:{item.status.lower()}", f"priority:{item.priority.lower()}"]
    if dry_run:
        print(f"DRY RUN: {item.title} [{', '.join(labels)}]")
        return

    run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            item.title,
            "--body",
            issue_body(item),
            "--label",
            ",".join(labels),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. lutzkuen/agentflow")
    parser.add_argument("--backlog", default="BACKLOG.md")
    parser.add_argument(
        "--statuses",
        default="READY,IDEA,BLOCKED,IN-PROGRESS",
        help="Comma-separated BACKLOG statuses to migrate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = {status.strip() for status in args.statuses.split(",") if status.strip()}
    items = [item for item in parse_backlog(Path(args.backlog)) if item.status in wanted]

    if not args.dry_run:
        ensure_label(args.repo, "backlog", "6f42c1", "Imported AgentFlow backlog item")
        for status in wanted:
            ensure_label(args.repo, f"status:{status.lower()}", "ededed", f"Backlog status {status}")
        for priority in sorted({item.priority for item in items}):
            ensure_label(args.repo, f"priority:{priority.lower()}", "1d76db", f"Backlog priority {priority}")

    existing = existing_issue_titles(args.repo) if not args.dry_run else set()
    created = 0
    skipped = 0
    for item in items:
        if item.title in existing:
            skipped += 1
            print(f"SKIP existing: {item.title}")
            continue
        create_issue(args.repo, item, args.dry_run)
        created += 1

    print(f"Backlog issue migration complete: created={created} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
