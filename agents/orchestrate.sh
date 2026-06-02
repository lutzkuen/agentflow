#!/usr/bin/env bash
# AgentFlow orchestrator entry point.
# Ensures proxy is up, then hands off to the Python orchestrator.

set -euo pipefail

export PATH="/home/lutz/.local/bin:$PATH"

REPO=/home/lutz/agentflow
cd "$REPO"
source .venv/bin/activate

# Ensure proxy is up
HEALTH=$(curl -sf http://localhost:4000/health 2>/dev/null || echo "UNREACHABLE")
if echo "$HEALTH" | grep -q "UNREACHABLE"; then
  echo "Proxy down — starting..."
  nohup python -m uvicorn agentflow_proxy.server:app \
    --host 127.0.0.1 --port 4000 > /tmp/agentflow.log 2>&1 &
  sleep 3
fi

python agents/run_orchestrator.py
