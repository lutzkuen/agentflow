#!/usr/bin/env bash
# Guarded AgentFlow orchestrator entry point.
# Sequence:
#   1. Ensure the supervised middleware is healthy.
#   2. Smoke test Claude OAuth through the middleware.
#   3. Run the Claude orchestrator.
#   4. Re-check and repair the middleware if Claude disturbed it.

set -Eeuo pipefail

export PATH="/home/lutz/.local/bin:$PATH"

REPO=${AGENTFLOW_REPO:-/home/lutz/agentflow}
LOG_DIR="$REPO/runs"
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M)}
PROXY_URL=${ANTHROPIC_BASE_URL:-http://127.0.0.1:4000}
PROXY_SERVICE=${AGENTFLOW_PROXY_SERVICE:-agentflow-claude-proxy.service}
CLAUDE_BIN=${CLAUDE_BIN:-claude}
SMOKE_EXPECT="AGENTFLOW_SMOKE_OK"
LOCK_FILE=${AGENTFLOW_ORCHESTRATOR_LOCK:-/tmp/agentflow-orchestrator.lock}
RUN_LOG="$LOG_DIR/$RUN_ID.md"

export RUN_ID
export ANTHROPIC_BASE_URL="$PROXY_URL"

mkdir -p "$LOG_DIR"
cd "$REPO"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another AgentFlow orchestrator run is active; exiting."
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

append_diagnostics() {
  local reason=$1
  {
    echo ""
    echo "## Guard Diagnostics"
    echo ""
    echo "Reason: $reason"
    echo ""
    echo "### Proxy Service"
    systemctl --user status "$PROXY_SERVICE" --no-pager || true
    echo ""
    echo "### Port 4000"
    ss -ltnp | grep -E '(:4000\\b|:4000 )' || true
    echo ""
    echo "### AgentFlow Processes"
    ps -eo pid,ppid,stat,cmd | grep -E 'uvicorn agentflow_proxy|agentflow_proxy.server|port 4000|port 4001' | grep -v grep || true
    echo ""
    echo "### Recent Proxy Journal"
    journalctl --user -u "$PROXY_SERVICE" -n 80 --no-pager || true
  } >> "$RUN_LOG"
}

wait_for_health() {
  local deadline=$((SECONDS + 30))
  until curl -sf "$PROXY_URL/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 1
  done
}

port_is_bound() {
  ss -ltnp | awk '$4 ~ /:4000$/' | grep -q .
}

repair_proxy_service() {
  log "Repairing supervised proxy service"
  systemctl --user daemon-reload
  systemctl --user reset-failed "$PROXY_SERVICE" >/dev/null 2>&1 || true

  if ! port_is_bound; then
    log "Reclaiming port 4000 from non-local or stale listener"
    fuser -k 4000/tcp >/dev/null 2>&1 || true
    sleep 1
  fi

  systemctl --user restart "$PROXY_SERVICE"
  wait_for_health
  port_is_bound
}

ensure_proxy_ready() {
  if curl -sf "$PROXY_URL/health" >/dev/null 2>&1 && port_is_bound; then
    return 0
  fi
  repair_proxy_service
}

claude_oauth_smoke() {
  local output
  output=$(
    timeout 90 env \
      -u ANTHROPIC_API_KEY \
      -u ANTHROPIC_AUTH_TOKEN \
      ANTHROPIC_BASE_URL="$PROXY_URL" \
      "$CLAUDE_BIN" --print --no-session-persistence --model haiku \
      "Reply with exactly: $SMOKE_EXPECT" 2>&1
  ) || {
    printf '%s\n' "$output"
    return 1
  }

  printf '%s\n' "$output" | grep -q "$SMOKE_EXPECT"
}

run_orchestrator() {
  {
    cat "$REPO/agents/orchestrator.md"
    echo ""
    echo "---"
    echo ""
    echo "# Live Context - $RUN_ID"
    echo ""
    echo "The recurring shell guard already verified Claude OAuth through $PROXY_URL before this run."
    echo "Do not stop, kill, restart, replace, or bind anything on prod port 4000."
    echo "Use dev port 4001 for development and tests. The shell guard owns prod health."
    echo ""
    echo "**Proxy health:** $(curl -sf "$PROXY_URL/health" 2>/dev/null)"
    echo ""
    echo "**Stats:**"
    curl -sf "$PROXY_URL/agentflow/stats" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "unavailable"
    echo ""
    echo "**Git log:**"
    git log --oneline -15
    echo ""
    echo "**BACKLOG.md:**"
    cat "$REPO/BACKLOG.md"
    echo ""
    echo "**Last run:**"
    ls -1t "$LOG_DIR"/*.md 2>/dev/null | grep -v "/$RUN_ID.md$" | head -1 | xargs tail -60 2>/dev/null || echo "(no previous runs)"
  } | env \
      -u ANTHROPIC_API_KEY \
      -u ANTHROPIC_AUTH_TOKEN \
      ANTHROPIC_BASE_URL="$PROXY_URL" \
      "$CLAUDE_BIN" --print --no-session-persistence \
      --append-system-prompt "Never kill, stop, restart, replace, or bind the AgentFlow prod middleware on port 4000. If prod appears broken, report it; the shell guard and Codex handle prod repair." \
      --allowedTools "Bash,Read,Write,Edit" \
      2>&1 | tee -a "$RUN_LOG"
}

main() {
  : > "$RUN_LOG"

  log "Preflight: ensuring proxy is healthy" | tee -a "$RUN_LOG"
  if ! ensure_proxy_ready; then
    log "Preflight failed: proxy could not be repaired by shell guard" | tee -a "$RUN_LOG"
    append_diagnostics "preflight proxy repair failed"
    echo "CODEX_REQUIRED: proxy preflight failed. Debug and repair the middleware before running Claude." | tee -a "$RUN_LOG"
    exit 20
  fi

  log "Preflight: smoke testing Claude OAuth through middleware" | tee -a "$RUN_LOG"
  if ! claude_oauth_smoke; then
    log "OAuth smoke failed; attempting one proxy repair and retry" | tee -a "$RUN_LOG"
    repair_proxy_service || true
    if ! claude_oauth_smoke; then
      append_diagnostics "preflight Claude OAuth smoke failed"
      echo "CODEX_REQUIRED: Claude OAuth smoke failed through middleware." | tee -a "$RUN_LOG"
      exit 21
    fi
  fi

  if [[ "${1:-}" == "--preflight-only" ]]; then
    log "Preflight-only mode passed" | tee -a "$RUN_LOG"
    echo "Run log: $RUN_LOG" | tee -a "$RUN_LOG"
    return 0
  fi

  log "Running Claude orchestrator" | tee -a "$RUN_LOG"
  if ! run_orchestrator; then
    log "Claude orchestrator exited non-zero; continuing to postflight middleware check" | tee -a "$RUN_LOG"
  fi

  log "Postflight: checking whether Claude disturbed the middleware" | tee -a "$RUN_LOG"
  if ! ensure_proxy_ready || ! claude_oauth_smoke; then
    log "Postflight failed; repairing and retrying smoke once" | tee -a "$RUN_LOG"
    repair_proxy_service || true
    if ! ensure_proxy_ready || ! claude_oauth_smoke; then
      append_diagnostics "postflight middleware or Claude OAuth smoke failed"
      echo "CODEX_REQUIRED: Claude disturbed the middleware and shell repair did not restore it." | tee -a "$RUN_LOG"
      exit 22
    fi
  fi

  log "Postflight smoke passed" | tee -a "$RUN_LOG"
  echo "" | tee -a "$RUN_LOG"
  echo "Run log: $RUN_LOG" | tee -a "$RUN_LOG"
}

main "$@"
