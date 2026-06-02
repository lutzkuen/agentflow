#!/usr/bin/env bash
# AgentFlow orchestrator entry point.
# Gathers context, then hands off entirely to the orchestrator agent.
# The orchestrator invokes sub-agents itself via its Bash tool.

set -euo pipefail

REPO=/home/lutz/agentflow
LOG_DIR="$REPO/runs"
RUN_ID=$(date +%Y-%m-%d_%H-%M)
export RUN_ID

mkdir -p "$LOG_DIR"
cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

# Ensure proxy is up before handing off
HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "UNREACHABLE")
if echo "$HEALTH" | grep -q "UNREACHABLE"; then
  nohup python -m uvicorn agentflow_proxy.server:app \
    --host 127.0.0.1 --port 4000 > /tmp/agentflow.log 2>&1 &
  sleep 3
fi

# Build context and invoke the orchestrator
{
  cat "$REPO/agents/orchestrator.md"
  echo ""
  echo "---"
  echo ""
  echo "# Live Context — $RUN_ID"
  echo ""
  echo "**Proxy health:** $(curl -sf http://localhost:4000/health 2>/dev/null || echo 'DOWN')"
  echo ""
  echo "**Stats:**"
  curl -sf http://localhost:4000/agentflow/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "unavailable"
  echo ""
  echo "**Git log:**"
  git log --oneline -15 2>/dev/null || echo "none"
  echo ""
  echo "**BACKLOG.md:**"
  cat "$REPO/BACKLOG.md"
  echo ""
  echo "**Last run:**"
  ls -1t "$LOG_DIR"/*.md 2>/dev/null | grep -v "$RUN_ID" | head -1 \
    | xargs tail -60 2>/dev/null || echo "(no previous runs)"
} | claude --print \
    --allowedTools "Bash,Read,Write,Edit" \
    2>&1 | tee "$LOG_DIR/$RUN_ID.md"

echo ""
echo "Run log: $LOG_DIR/$RUN_ID.md"
