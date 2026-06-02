#!/usr/bin/env bash
# AgentFlow orchestrator — explicit multi-agent pipeline.
#
# Phases run sequentially, each a separate claude invocation:
#   1. planner   — reads state, decides what to work on, outputs JSON
#   2. worker    — implements the plan with full tool access
#   3. tester    — validates the proxy still works
#   4. committer — commits if tests passed
#   5. scribe    — writes the run summary

set -euo pipefail

REPO=/home/lutz/agentflow
LOG_DIR="$REPO/runs"
RUN_ID=$(date +%Y-%m-%d_%H-%M)
RUN_LOG="$LOG_DIR/$RUN_ID.md"
WORK_DIR=$(mktemp -d /tmp/agentflow-run-XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$LOG_DIR"
cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

log() { echo "$*" | tee -a "$RUN_LOG"; }

log "# AgentFlow Run: $RUN_ID"
log ""

# ── Ensure proxy is up ───────────────────────────────────────────────────────

HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "UNREACHABLE")
if echo "$HEALTH" | grep -q "UNREACHABLE"; then
  log "## Proxy was down — restarting"
  nohup python -m uvicorn agentflow_proxy.server:app \
    --host 127.0.0.1 --port 4000 > /tmp/agentflow.log 2>&1 &
  sleep 3
  HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "STILL_DOWN")
fi
log "Health: $HEALTH"
log ""

# ── Collect context for planner ──────────────────────────────────────────────

STATS=$(curl -sf http://localhost:4000/agentflow/stats 2>/dev/null \
  | python3 -m json.tool 2>/dev/null || echo "unavailable")
GIT_LOG=$(git log --oneline -15 2>/dev/null || echo "none")
LAST_RUN_SUMMARY=""
LAST_RUN=$(ls -1t "$LOG_DIR"/*.md 2>/dev/null | grep -v "$RUN_ID" | head -1 || true)
if [ -n "$LAST_RUN" ]; then
  LAST_RUN_SUMMARY=$(tail -80 "$LAST_RUN")
fi

cat > "$WORK_DIR/context.md" << EOF
# AgentFlow State — $RUN_ID

## Proxy health
$HEALTH

## Stats
$STATS

## Git log (last 15)
$GIT_LOG

## Backlog
$(cat "$REPO/BACKLOG.md")

## Last run summary
${LAST_RUN_SUMMARY:-"(no previous runs)"}
EOF

# ── Phase 1: Planner ─────────────────────────────────────────────────────────
# Pure reasoning — no tools. Reads context, outputs a specific actionable task.

log "## Phase 1: Planner"

cat > "$WORK_DIR/planner_prompt.md" << 'EOF'
You are the AgentFlow planner. You decide what the developer agent should work on this run.

Read the context carefully, then output a JSON task decision.
Be specific: a vague task wastes the developer's time.

Rules:
- Pick the single highest-value READY item from the backlog.
- If the last run failed or was blocked, prioritize unblocking that.
- If all P0 items are DONE, move to P1.
- If no READY items exist, output action=analyze so we find new work.
- Prefer items that are small enough to complete in one agent run (~30 min of coding).

Output ONLY this JSON (no other text):
{
  "action": "develop" | "analyze" | "research" | "test",
  "item": "<exact backlog item title>",
  "rationale": "<one sentence why this is the right choice>",
  "implementation_hint": "<specific approach: what file, what function, what to add>"
}
EOF

PLAN=$( (cat "$WORK_DIR/planner_prompt.md"; echo; echo "---"; cat "$WORK_DIR/context.md") \
  | claude --print 2>/dev/null )

log "Plan: $PLAN"
log ""

# Parse plan
ACTION=$(echo "$PLAN" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['action'])" 2>/dev/null || echo "analyze")
ITEM=$(echo "$PLAN" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['item'])" 2>/dev/null || echo "unknown")
HINT=$(echo "$PLAN" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('implementation_hint',''))" 2>/dev/null || echo "")

# ── Phase 2: Worker ──────────────────────────────────────────────────────────
# Full tool access. Actually implements the plan.

log "## Phase 2: Worker (action=$ACTION, item=$ITEM)"

WORKER_PROMPT_FILE="$REPO/agents/develop.md"
if [ "$ACTION" = "analyze" ]; then
  WORKER_PROMPT_FILE="$REPO/agents/analyze.md"
elif [ "$ACTION" = "research" ]; then
  WORKER_PROMPT_FILE="$REPO/agents/research.md"
elif [ "$ACTION" = "test" ]; then
  WORKER_PROMPT_FILE="$REPO/agents/test.md"
fi

cat > "$WORK_DIR/worker_prompt.md" << EOF
$(cat "$WORKER_PROMPT_FILE")

---

# Your Specific Task This Run

**Item:** $ITEM
**Implementation hint:** $HINT

You are in /home/lutz/agentflow. The proxy is running on port 4000.
Do the work, test it, and stop. Do not plan future work — just implement this one item.
EOF

claude --print \
  --allowedTools "Bash,Read,Write,Edit" \
  < "$WORK_DIR/worker_prompt.md" \
  2>&1 | tee "$WORK_DIR/worker_output.txt"

log ""
log "### Worker output"
cat "$WORK_DIR/worker_output.txt" | tee -a "$RUN_LOG"
log ""

# ── Phase 3: Tester ──────────────────────────────────────────────────────────
# Validates the proxy. Independent of what the worker did.

log "## Phase 3: Tester"

cat > "$WORK_DIR/tester_prompt.md" << EOF
$(cat "$REPO/agents/test.md")

---

# Context
The worker agent just implemented: $ITEM

Run all the tests. At the very end, output one line:
VERDICT: PASS
or
VERDICT: FAIL — <reason>
EOF

TEST_OUTPUT=$(claude --print \
  --allowedTools "Bash,Read" \
  < "$WORK_DIR/tester_prompt.md" \
  2>&1)

echo "$TEST_OUTPUT" | tee -a "$RUN_LOG"
log ""

VERDICT=$(echo "$TEST_OUTPUT" | grep "^VERDICT:" | tail -1 || echo "VERDICT: UNKNOWN")
log "Tester verdict: $VERDICT"
log ""

# ── Phase 4: Commit ──────────────────────────────────────────────────────────

log "## Phase 4: Commit"

if echo "$VERDICT" | grep -q "PASS"; then
  CHANGED=$(git status --short | grep -v "^??" | wc -l || echo 0)
  if [ "$CHANGED" -gt 0 ]; then
    git add -A
    git commit -m "agent: $ITEM

Implemented by orchestrator run $RUN_ID.
Action: $ACTION
Verdict: $VERDICT"
    log "Committed: $ITEM"
  else
    log "No file changes to commit (analysis/research run, or no-op)."
  fi
else
  log "Skipping commit — tests did not pass. Changes left unstaged."
fi
log ""

# ── Phase 5: Scribe ──────────────────────────────────────────────────────────
# Summarizes what happened and updates BACKLOG.md.

log "## Phase 5: Scribe"

cat > "$WORK_DIR/scribe_prompt.md" << EOF
You are the AgentFlow scribe. Read what just happened and:
1. Update BACKLOG.md: mark the completed item DONE (if tests passed), or add a BLOCKED note.
2. Write a concise summary to stdout (will be appended to the run log).

# What happened this run

Run ID: $RUN_ID
Action taken: $ACTION
Item: $ITEM
Test verdict: $VERDICT

## Worker output
$(cat "$WORK_DIR/worker_output.txt" | head -100)

## Current BACKLOG.md
$(cat "$REPO/BACKLOG.md")

---

Instructions:
- If VERDICT: PASS and action=develop: change the item's status from READY to DONE with today's date.
- If VERDICT: FAIL: change the item's status to BLOCKED with a note about what failed.
- If action=analyze or research: append any new findings to the "Agent Findings" section.
- Then output a 10-15 line run summary (what was done, verdict, what the next run should focus on).
EOF

SUMMARY=$(claude --print \
  --allowedTools "Read,Write,Edit" \
  < "$WORK_DIR/scribe_prompt.md" \
  2>&1)

log "### Run Summary"
echo "$SUMMARY" | tee -a "$RUN_LOG"
log ""
log "--- run complete: $RUN_ID ---"

echo ""
echo "Run log: $RUN_LOG"
