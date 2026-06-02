#!/usr/bin/env bash
# AgentFlow orchestrator entry point.
# Collects context, pipes it into the orchestrator agent.
# All API calls route through the proxy via ANTHROPIC_BASE_URL.

set -euo pipefail

export PATH="/home/lutz/.local/bin:$PATH"

REPO=/home/lutz/agentflow
LOG_DIR="$REPO/runs"
RUN_ID=$(date +%Y-%m-%d_%H-%M)
export RUN_ID

mkdir -p "$LOG_DIR"
cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

# Ensure proxy is up
if ! curl -sf http://localhost:4000/health > /dev/null 2>&1; then
  echo "Proxy down — starting..."
  nohup python -m uvicorn agentflow_proxy.server:app \
    --host 127.0.0.1 --port 4000 > /tmp/agentflow.log 2>&1 &
  sleep 3
fi

# Pipe orchestrator prompt + live context into claude
{
  cat "$REPO/agents/orchestrator.md"
  echo ""
  echo "---"
  echo ""
  echo "# Live Context — $RUN_ID"
  echo ""
  echo "**Proxy health:** $(curl -sf http://localhost:4000/health 2>/dev/null)"
  echo ""
  echo "**Stats:**"
  curl -sf http://localhost:4000/agentflow/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "unavailable"
  echo ""
  echo "**Git log:**"
  git log --oneline -15
  echo ""
  echo "**BACKLOG.md:**"
  cat "$REPO/BACKLOG.md"
  echo ""
  echo "**Last run:**"
  ls -1t "$LOG_DIR"/*.md 2>/dev/null | head -1 | xargs tail -60 2>/dev/null || echo "(no previous runs)"
} | claude --print \
    --allowedTools "Bash,Read,Write,Edit" \
    2>&1 | tee "$LOG_DIR/$RUN_ID.md"

echo ""
echo "Run log: $LOG_DIR/$RUN_ID.md"
