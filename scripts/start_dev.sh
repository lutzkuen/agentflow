#!/usr/bin/env bash
# Start the dev proxy on port 4001 with a separate DB.
# Safe to run while prod (port 4000) is live.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=${AGENTFLOW_VENV:-/home/lutz/agentflow/.venv}

# Kill any existing port 4001 process
pkill -f "uvicorn agentflow_proxy.*4001" 2>/dev/null || true
sleep 1

# Ensure dev DB directory exists
mkdir -p "$HOME/.agentflow"

cd "$REPO"
source "$VENV/bin/activate"

nohup env AGENTFLOW_PORT=4001 AGENTFLOW_DB="$HOME/.agentflow/dev.sqlite3" \
    python -m uvicorn agentflow_proxy.server:app --host 127.0.0.1 --port 4001 \
    > /tmp/agentflow-dev.log 2>&1 &

sleep 2

if curl -sf http://localhost:4001/health > /dev/null; then
    echo "Dev proxy running on port 4001 (DB: $HOME/.agentflow/dev.sqlite3)"
else
    echo "ERROR: dev proxy failed to start. Check /tmp/agentflow-dev.log"
    exit 1
fi
