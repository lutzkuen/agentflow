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
CLAUDE_PROJECT_DIR=${CLAUDE_PROJECT_DIR:-$HOME/.claude/projects/-home-lutz-agentflow}
SMOKE_EXPECT="AGENTFLOW_SMOKE_OK"
LOCK_FILE=${AGENTFLOW_ORCHESTRATOR_LOCK:-/tmp/agentflow-orchestrator.lock}
RUN_LOG="$LOG_DIR/$RUN_ID.md"
CLAUDE_SMOKE_OUTPUT=""

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

claude_output_is_transient_rate_limit() {
  grep -Eiq 'temporarily limiting requests|account.s rate limit|rate limit.*try again later'
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
    CLAUDE_SMOKE_OUTPUT="$output"
    printf '%s\n' "$output"
    if printf '%s\n' "$output" | claude_output_is_transient_rate_limit; then
      return 2
    fi
    return 1
  }

  CLAUDE_SMOKE_OUTPUT="$output"
  printf '%s\n' "$output" | grep -q "$SMOKE_EXPECT"
}

run_log_has_transient_rate_limit() {
  tail -120 "$RUN_LOG" 2>/dev/null | claude_output_is_transient_rate_limit
}

capture_claude_oauth_smoke() {
  set +e
  claude_oauth_smoke
  smoke_status=$?
  set -e
}

find_broken_token_session() {
  python3 - "$CLAUDE_PROJECT_DIR" <<'PY'
import glob
import json
import os
import re
import sys

project_dir = sys.argv[1]
patterns = [
    r"out of tokens",
    r"too many tokens",
    r"token limit",
    r"maximum[^\n]{0,80}tokens?",
    r"context window",
    r"context length",
    r"prompt is too long",
    r"input is too long",
    r"exceeds?[^\n]{0,80}context",
    r"session limit",
    r"usage limit",
    r"you'?ve hit your session limit",
    r"rate[_ -]?limit",
]
problem_re = re.compile("|".join(patterns), re.IGNORECASE)


def flatten_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key in ("type", "text", "message", "content", "error", "status"):
            if key in value:
                parts.append(flatten_text(value[key]))
        if value.get("isApiErrorMessage"):
            parts.append("api error")
        return "\n".join(parts)
    return str(value)


def file_candidate(path):
    session_id = os.path.splitext(os.path.basename(path))[0]
    last_problem = None
    last_success = None
    reason = ""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"raw": line}

                session_id = event.get("sessionId") or session_id
                text = flatten_text(event)
                match = problem_re.search(text)
                if match:
                    last_problem = index
                    reason = match.group(0)

                message = event.get("message") if isinstance(event, dict) else None
                if isinstance(message, dict):
                    if message.get("role") == "assistant" and message.get("stop_reason") == "end_turn":
                        last_success = index
                    content_text = flatten_text(message.get("content"))
                    if event.get("type") == "assistant" and message.get("role") == "assistant" and content_text:
                        if not problem_re.search(content_text) and not event.get("isApiErrorMessage"):
                            last_success = index
    except OSError:
        return None

    if last_problem is None:
        return None
    if last_success is not None and last_success > last_problem:
        return None
    return (os.path.getmtime(path), session_id, reason, path)


candidates = []
for path in glob.glob(os.path.join(project_dir, "*.jsonl")):
    candidate = file_candidate(path)
    if candidate:
        candidates.append(candidate)

if not candidates:
    raise SystemExit(1)

candidates.sort(reverse=True)
_, session_id, reason, path = candidates[0]
print(session_id)
print(f"Selected unresolved Claude token/session-limit session {session_id} ({reason}) from {path}", file=sys.stderr)
PY
}

write_orchestrator_prompt() {
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
  }
}

run_claude_operator() {
  local resume_session=${1:-}
  local -a claude_args
  claude_args=(
    --print
    --append-system-prompt
    "Never kill, stop, restart, replace, or bind the AgentFlow prod middleware on port 4000. If prod appears broken, report it; the shell guard and Codex handle prod repair. If this run was started with --resume, inspect the existing session context first and continue the unfinished work instead of repeating completed work."
    --allowedTools
    "Bash,Read,Write,Edit"
  )

  if [[ -n "$resume_session" ]]; then
    claude_args+=(--resume "$resume_session")
  fi

  write_orchestrator_prompt | env \
      -u ANTHROPIC_API_KEY \
      -u ANTHROPIC_AUTH_TOKEN \
      ANTHROPIC_BASE_URL="$PROXY_URL" \
      "$CLAUDE_BIN" "${claude_args[@]}" \
      2>&1 | tee -a "$RUN_LOG"
}

run_orchestrator() {
  local resume_session
  resume_session=$(find_broken_token_session 2>>"$RUN_LOG" || true)

  if [[ -n "$resume_session" ]]; then
    log "Running Claude orchestrator by resuming token-broken session $resume_session" | tee -a "$RUN_LOG"
    run_claude_operator "$resume_session"
  else
    log "No unresolved token-broken Claude session found; starting a new operator session" | tee -a "$RUN_LOG"
    run_claude_operator
  fi
}

main() {
  if [[ "${1:-}" == "--print-resume-candidate" ]]; then
    find_broken_token_session
    return $?
  fi

  : > "$RUN_LOG"

  log "Preflight: ensuring proxy is healthy" | tee -a "$RUN_LOG"
  if ! ensure_proxy_ready; then
    log "Preflight failed: proxy could not be repaired by shell guard" | tee -a "$RUN_LOG"
    append_diagnostics "preflight proxy repair failed"
    echo "CODEX_REQUIRED: proxy preflight failed. Debug and repair the middleware before running Claude." | tee -a "$RUN_LOG"
    exit 20
  fi

  log "Preflight: smoke testing Claude OAuth through middleware" | tee -a "$RUN_LOG"
  capture_claude_oauth_smoke
  if (( smoke_status == 2 )); then
    log "Claude upstream is temporarily rate-limiting requests; middleware is healthy, skipping repair" | tee -a "$RUN_LOG"
    echo "CLAUDE_RATE_LIMITED: try this run again later." | tee -a "$RUN_LOG"
    exit 23
  fi
  if (( smoke_status != 0 )); then
    log "OAuth smoke failed; attempting one proxy repair and retry" | tee -a "$RUN_LOG"
    repair_proxy_service || true
    capture_claude_oauth_smoke
    if (( smoke_status == 2 )); then
      log "Claude upstream is temporarily rate-limiting requests after repair; middleware is healthy" | tee -a "$RUN_LOG"
      echo "CLAUDE_RATE_LIMITED: try this run again later." | tee -a "$RUN_LOG"
      exit 23
    fi
    if (( smoke_status != 0 )); then
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
    if run_log_has_transient_rate_limit; then
      log "Claude orchestrator hit transient upstream rate limiting; checking proxy health only" | tee -a "$RUN_LOG"
      if ensure_proxy_ready; then
        echo "CLAUDE_RATE_LIMITED: middleware is healthy; try this run again later." | tee -a "$RUN_LOG"
        exit 23
      fi
    fi
    log "Claude orchestrator exited non-zero; continuing to postflight middleware check" | tee -a "$RUN_LOG"
  fi

  log "Postflight: checking whether Claude disturbed the middleware" | tee -a "$RUN_LOG"
  if ! ensure_proxy_ready; then
    log "Postflight failed; repairing and retrying smoke once" | tee -a "$RUN_LOG"
    repair_proxy_service || true
    if ! ensure_proxy_ready; then
      append_diagnostics "postflight middleware or Claude OAuth smoke failed"
      echo "CODEX_REQUIRED: Claude disturbed the middleware and shell repair did not restore it." | tee -a "$RUN_LOG"
      exit 22
    fi
  fi
  capture_claude_oauth_smoke
  if (( smoke_status == 2 )); then
    log "Postflight proxy is healthy; Claude upstream is temporarily rate-limiting smoke requests" | tee -a "$RUN_LOG"
    echo "CLAUDE_RATE_LIMITED: middleware is healthy; try this run again later." | tee -a "$RUN_LOG"
    exit 23
  fi
  if (( smoke_status != 0 )); then
    log "Postflight smoke failed; repairing and retrying smoke once" | tee -a "$RUN_LOG"
    repair_proxy_service || true
    capture_claude_oauth_smoke
    if (( smoke_status == 2 )); then
      log "Postflight proxy is healthy after repair; Claude upstream is temporarily rate-limiting smoke requests" | tee -a "$RUN_LOG"
      echo "CLAUDE_RATE_LIMITED: middleware is healthy; try this run again later." | tee -a "$RUN_LOG"
      exit 23
    fi
    if (( smoke_status != 0 )); then
      append_diagnostics "postflight Claude OAuth smoke failed"
      echo "CODEX_REQUIRED: Claude OAuth smoke failed through middleware." | tee -a "$RUN_LOG"
      exit 22
    fi
  fi

  log "Postflight smoke passed" | tee -a "$RUN_LOG"
  echo "" | tee -a "$RUN_LOG"
  echo "Run log: $RUN_LOG" | tee -a "$RUN_LOG"
}

main "$@"
