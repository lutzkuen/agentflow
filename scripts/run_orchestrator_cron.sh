#!/usr/bin/env bash
# Cron wrapper for unattended AgentFlow orchestrator runs.

set -uo pipefail

REPO=${AGENTFLOW_REPO:-/home/lutz/agentflow}
LOG_DIR="$REPO/logs/orchestrator"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="$LOG_DIR/$STAMP.log"
LATEST_LOG="$LOG_DIR/latest.log"
status=0

mkdir -p "$LOG_DIR"

{
  echo "## AgentFlow Cron Run"
  echo "Started: $(date --iso-8601=seconds)"
  echo "Host: $(hostname)"
  echo "Repo: $REPO"
  echo "Run ID: ${RUN_ID:-$STAMP}"
  echo ""

  if ! cd "$REPO"; then
    status=70
  else
    export PATH="/home/lutz/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
    export HOME=${HOME:-/home/lutz}
    export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
    export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
    export ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-http://127.0.0.1:4000}
    export RUN_ID=${RUN_ID:-$STAMP}

    ./agents/orchestrate.sh
    status=$?
  fi

  echo ""
  echo "Finished: $(date --iso-8601=seconds)"
  echo "Exit: $status"
} > "$LOG_FILE" 2>&1

if (( status != 0 )) && grep -q "CODEX_REQUIRED" "$LOG_FILE"; then
  recovery_status=0
  {
    echo ""
    echo "## Codex Recovery"
    echo "Started: $(date --iso-8601=seconds)"
    AGENTFLOW_REPO="$REPO" AGENTFLOW_FAILURE_LOG="$LOG_FILE" \
      "$REPO/scripts/codex_recover.sh" "$LOG_FILE"
    recovery_status=$?
    echo "Finished: $(date --iso-8601=seconds)"
    echo "Exit: $recovery_status"
  } >> "$LOG_FILE" 2>&1
  if (( recovery_status == 0 )); then
    status=0
  fi
fi

ln -sfn "$LOG_FILE" "$LATEST_LOG"
exit "$status"
