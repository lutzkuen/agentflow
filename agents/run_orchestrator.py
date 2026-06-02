#!/usr/bin/env python3
"""
AgentFlow orchestrator — Claude-native multi-agent loop.

The orchestrator is an Anthropic API call in a tool-use loop.
Sub-agents are tools the orchestrator calls. Each tool spins up a focused
`claude --print` process and returns its output as a tool result.
The model decides which agents to invoke and in what order.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

REPO = Path(__file__).parent.parent
AGENTS_DIR = REPO / "agents"
RUNS_DIR = REPO / "runs"
RUNS_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
RUN_LOG = RUNS_DIR / f"{RUN_ID}.md"

client = anthropic.Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:4000"),
    api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN", "proxy-passthrough"),
)

# ── Sub-agent tool definitions ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_developer",
        "description": (
            "Invoke the developer sub-agent to implement a specific backlog item. "
            "The agent edits code in the repo, restarts the proxy, and verifies the change. "
            "Returns the agent's full output including what was changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Exact backlog item title to implement"},
                "hint": {"type": "string", "description": "Specific implementation approach / files to touch"},
            },
            "required": ["item"],
        },
    },
    {
        "name": "run_tester",
        "description": (
            "Invoke the test sub-agent to validate the proxy is working correctly. "
            "Runs health check, cache, routing, streaming, and stats tests. "
            "Returns full output ending with 'VERDICT: PASS' or 'VERDICT: FAIL — reason'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_analyzer",
        "description": (
            "Invoke the analysis sub-agent to query the SQLite DB and find optimization "
            "opportunities. Returns findings and appends new IDEA items to BACKLOG.md."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_researcher",
        "description": (
            "Invoke the research sub-agent to find new crunching/routing/caching techniques. "
            "Returns findings and appends them to BACKLOG.md."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "commit_changes",
        "description": "Commit all current changes to git. Only call this after tests pass.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "update_backlog_item",
        "description": "Update the status of a backlog item in BACKLOG.md.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Exact item title"},
                "status": {
                    "type": "string",
                    "enum": ["IN-PROGRESS", "DONE", "BLOCKED"],
                    "description": "New status",
                },
                "note": {"type": "string", "description": "Optional note to append (e.g. done date or block reason)"},
            },
            "required": ["item", "status"],
        },
    },
    {
        "name": "write_run_summary",
        "description": "Write the final summary for this orchestrator run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Multi-line markdown summary of what was done"},
            },
            "required": ["summary"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def _run_subagent(prompt_file: Path, extra: str = "", allowed_tools: str = "Bash,Read,Write,Edit") -> str:
    prompt = prompt_file.read_text()
    if extra:
        prompt = prompt + "\n\n---\n\n" + extra
    result = subprocess.run(
        ["claude", "--print", f"--allowedTools={allowed_tools}"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
    )
    return (result.stdout + result.stderr).strip()


def tool_run_developer(item: str, hint: str = "") -> str:
    extra = f"# Your Task This Run\n\n**Item:** {item}\n**Hint:** {hint or 'See backlog for details.'}"
    print(f"  → developer: {item}", flush=True)
    return _run_subagent(AGENTS_DIR / "develop.md", extra=extra)


def tool_run_tester() -> str:
    print("  → tester", flush=True)
    return _run_subagent(AGENTS_DIR / "test.md", allowed_tools="Bash,Read")


def tool_run_analyzer() -> str:
    print("  → analyzer", flush=True)
    return _run_subagent(AGENTS_DIR / "analyze.md")


def tool_run_researcher() -> str:
    print("  → researcher", flush=True)
    return _run_subagent(AGENTS_DIR / "research.md")


def tool_commit_changes(message: str) -> str:
    r1 = subprocess.run(["git", "add", "-A"], cwd=REPO, capture_output=True, text=True)
    r2 = subprocess.run(["git", "commit", "-m", message], cwd=REPO, capture_output=True, text=True)
    return (r2.stdout + r2.stderr).strip() or "nothing to commit"


def tool_update_backlog_item(item: str, status: str, note: str = "") -> str:
    backlog = (REPO / "BACKLOG.md").read_text()
    date_str = datetime.now().strftime("%Y-%m-%d")
    suffix = f" ({date_str}{': ' + note if note else ''})"
    # Replace the status tag for the matching line
    import re
    pattern = re.compile(r"(- \[)[A-Z-]+(]\s+" + re.escape(item) + r")")
    new_backlog, count = pattern.subn(r"\g<1>" + status + r"\2" + suffix, backlog)
    if count:
        (REPO / "BACKLOG.md").write_text(new_backlog)
        return f"Updated '{item}' to {status}."
    return f"Item '{item}' not found in BACKLOG.md — no change made."


def tool_write_run_summary(summary: str) -> str:
    RUN_LOG.write_text(f"# AgentFlow Run: {RUN_ID}\n\n{summary}\n")
    return f"Summary written to {RUN_LOG}"


TOOL_DISPATCH = {
    "run_developer": lambda i: tool_run_developer(i["item"], i.get("hint", "")),
    "run_tester": lambda _: tool_run_tester(),
    "run_analyzer": lambda _: tool_run_analyzer(),
    "run_researcher": lambda _: tool_run_researcher(),
    "commit_changes": lambda i: tool_commit_changes(i["message"]),
    "update_backlog_item": lambda i: tool_update_backlog_item(i["item"], i["status"], i.get("note", "")),
    "write_run_summary": lambda i: tool_write_run_summary(i["summary"]),
}


# ── Orchestrator loop ─────────────────────────────────────────────────────────

def build_context() -> str:
    health = subprocess.run(
        ["curl", "-sf", "http://localhost:4000/health"],
        capture_output=True, text=True
    ).stdout or "UNREACHABLE"

    stats = subprocess.run(
        ["curl", "-sf", "http://localhost:4000/agentflow/stats"],
        capture_output=True, text=True
    ).stdout or "unavailable"

    git_log = subprocess.run(
        ["git", "log", "--oneline", "-15"],
        capture_output=True, text=True, cwd=REPO
    ).stdout or "none"

    backlog = (REPO / "BACKLOG.md").read_text()

    last_runs = sorted(RUNS_DIR.glob("*.md"), reverse=True)
    last_run = ""
    for r in last_runs:
        if r.name != RUN_LOG.name:
            last_run = r.read_text()[-3000:]
            break

    return f"""# Live Context — {RUN_ID}

**Proxy health:** {health}

**Stats:**
{stats}

**Git log (last 15):**
{git_log}

**BACKLOG.md:**
{backlog}

**Last run summary:**
{last_run or "(no previous runs)"}
"""


SYSTEM_PROMPT = (AGENTS_DIR / "orchestrator.md").read_text()


def run() -> None:
    context = build_context()
    messages = [{"role": "user", "content": context}]

    print(f"Run: {RUN_ID}", flush=True)

    while True:
        response = client.messages.create(
            model=os.getenv("AGENTFLOW_ORCHESTRATOR_MODEL", "claude-sonnet-4.5"),
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Print any text the orchestrator outputs
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text, flush=True)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n[tool: {block.name}]", flush=True)
                    try:
                        result = TOOL_DISPATCH[block.name](block.input)
                    except Exception as exc:
                        result = f"ERROR: {exc}"
                    print(result[:500], flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break


if __name__ == "__main__":
    run()
