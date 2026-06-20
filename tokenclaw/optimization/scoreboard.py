from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence

from tokenclaw.optimization.cli_support import default_db_path, open_store_for_db, write_json


def openai_scoreboard_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report whether OpenAI optimizations are helping from local metadata")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from tokenclaw.stats import stats_openai_scoreboard

    store = open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_openai_scoreboard(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0
