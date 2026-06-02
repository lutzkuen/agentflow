#!/usr/bin/env bash
# AgentFlow orchestrator entry point.
# Invoked every 2 hours by systemd timer or cron.
# Gathers context, then hands off to the orchestrator Claude agent.

set -euo pipefail

REPO=/home/lutz/agentflow
LOG_DIR="$REPO/runs"
RUN_ID=$(date +%Y-%m-%d_%H-%M)
RUN_LOG="$LOG_DIR/$RUN_ID.md"

mkdir -p "$LOG_DIR"

cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

echo "=== AgentFlow Orchestrator Run: $RUN_ID ===" | tee "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

# ── Gather context ──────────────────────────────────────────────────────────

echo "## Proxy Health" | tee -a "$RUN_LOG"
HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "UNREACHABLE")
echo "$HEALTH" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

# Start proxy if not running
if echo "$HEALTH" | grep -q "UNREACHABLE"; then
  echo "## Starting proxy (was not running)" | tee -a "$RUN_LOG"
  nohup python -m uvicorn agentflow_proxy.server:app \
    --host 127.0.0.1 --port 4000 \
    > /tmp/agentflow.log 2>&1 &
  sleep 3
  HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "STILL_UNREACHABLE")
  echo "$HEALTH" | tee -a "$RUN_LOG"
  echo "" | tee -a "$RUN_LOG"
fi

echo "## Recent Stats" | tee -a "$RUN_LOG"
STATS=$(curl -sf http://localhost:4000/agentflow/stats 2>/dev/null \
  | python3 -m json.tool 2>/dev/null || echo "stats unavailable")
echo "$STATS" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

echo "## Git Log (last 10)" | tee -a "$RUN_LOG"
git log --oneline -10 2>/dev/null | tee -a "$RUN_LOG" || echo "(no commits yet)" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

echo "## Last Run" | tee -a "$RUN_LOG"
LAST_RUN=$(ls -1t "$LOG_DIR"/*.md 2>/dev/null | grep -v "$RUN_ID" | head -1)
if [ -n "$LAST_RUN" ]; then
  echo "File: $LAST_RUN" | tee -a "$RUN_LOG"
  tail -40 "$LAST_RUN" | tee -a "$RUN_LOG"
else
  echo "(no previous runs)" | tee -a "$RUN_LOG"
fi
echo "" | tee -a "$RUN_LOG"

# ── Invoke orchestrator agent ───────────────────────────────────────────────

echo "## Orchestrator Agent Output" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

# Build prompt with live context injected
{
  cat "$REPO/agents/orchestrator.md"
  echo ""
  echo "---"
  echo ""
  echo "# Live Context for This Run"
  echo ""
  echo "Run ID: $RUN_ID"
  echo ""
  echo "### Proxy health"
  echo "$HEALTH"
  echo ""
  echo "### Stats"
  echo "$STATS"
  echo ""
  echo "### Git log"
  git log --oneline -10 2>/dev/null || echo "(none)"
  echo ""
  echo "### Current BACKLOG.md"
  cat "$REPO/BACKLOG.md"
  echo ""
  if [ -n "$LAST_RUN" ]; then
    echo "### Last run summary ($LAST_RUN)"
    tail -60 "$LAST_RUN"
  fi
} | claude --print \
    --allowedTools "Bash,Read,Write,Edit" \
    2>&1 | tee -a "$RUN_LOG"

echo "" | tee -a "$RUN_LOG"
echo "--- run complete: $RUN_ID ---" | tee -a "$RUN_LOG"

echo ""
echo "Run log written to: $RUN_LOG"
