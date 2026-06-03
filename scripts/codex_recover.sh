#!/usr/bin/env bash
# Run a bounded Codex salvage pass after AgentFlow cron reports CODEX_REQUIRED.

set -Eeuo pipefail

REPO=${AGENTFLOW_REPO:-/home/lutz/agentflow}
FAILURE_LOG=${1:-${AGENTFLOW_FAILURE_LOG:-}}
DEFAULT_CODEX_BIN=$(command -v codex || true)
if [[ -z "$DEFAULT_CODEX_BIN" && -x "$HOME/.vscode/extensions/openai.chatgpt-26.527.60818-linux-x64/bin/linux-x86_64/codex" ]]; then
  DEFAULT_CODEX_BIN="$HOME/.vscode/extensions/openai.chatgpt-26.527.60818-linux-x64/bin/linux-x86_64/codex"
fi
CODEX_BIN=${CODEX_BIN:-$DEFAULT_CODEX_BIN}
RECOVERY_DIR="$REPO/logs/codex-recovery"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
RECOVERY_LOG="$RECOVERY_DIR/$STAMP.log"
SUMMARY_LOG="$RECOVERY_DIR/$STAMP.summary"
LATEST_LOG="$RECOVERY_DIR/latest.log"
LAST_ATTEMPT="$RECOVERY_DIR/last_attempt"
COOLDOWN_MINUTES=${AGENTFLOW_CODEX_RECOVERY_COOLDOWN_MINUTES:-180}
CODEX_AUTO=${AGENTFLOW_CODEX_AUTO:-1}
CODEX_MODEL=${AGENTFLOW_CODEX_MODEL:-}
status=0

mkdir -p "$RECOVERY_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$CODEX_AUTO" != "1" ]]; then
  log "Codex auto-recovery disabled (AGENTFLOW_CODEX_AUTO=$CODEX_AUTO)"
  exit 75
fi

if [[ -z "$FAILURE_LOG" || ! -f "$FAILURE_LOG" ]]; then
  log "No failure log provided; skipping Codex recovery"
  exit 64
fi

if ! grep -q "CODEX_REQUIRED" "$FAILURE_LOG"; then
  log "Failure log does not contain CODEX_REQUIRED; skipping"
  exit 0
fi

if [[ -z "$CODEX_BIN" || ! -x "$CODEX_BIN" ]]; then
  log "Codex CLI not found; skipping recovery"
  exit 69
fi

if [[ -f "$LAST_ATTEMPT" ]]; then
  now=$(date +%s)
  last=$(stat -c %Y "$LAST_ATTEMPT")
  elapsed=$((now - last))
  cooldown=$((COOLDOWN_MINUTES * 60))
  if (( elapsed < cooldown )); then
    log "Codex recovery attempted ${elapsed}s ago; cooldown is ${cooldown}s"
    exit 75
  fi
fi
touch "$LAST_ATTEMPT"

{
  echo "## AgentFlow Codex Recovery"
  echo "Started: $(date --iso-8601=seconds)"
  echo "Repo: $REPO"
  echo "Failure log: $FAILURE_LOG"
  echo "Codex: $CODEX_BIN"
  echo ""

  cd "$REPO"

  prompt=$(cat <<PROMPT
You are Codex running unattended AgentFlow recovery.

Goal: inspect the CODEX_REQUIRED failure in this log and recover the repository to a clean,
ready-for-cron state:

$FAILURE_LOG

Rules:
- Do not start new feature work.
- If there is an unresolved AgentFlow run worktree, inspect it, validate its changes, finish
  only the small leftover bookkeeping if safe, commit, fast-forward merge into master, then
  remove the recovered worktree and branch.
- If the issue is proxy health, debug and repair the middleware conservatively. Restart the
  live proxy only when needed for health recovery.
- Run focused checks before merging or declaring recovery complete.
- Leave main clean. Do not use destructive commands like git reset --hard.
- If recovery is unsafe, write a clear note to runs/CODEX_REQUIRED.md with the blocker,
  preserved branch/worktree, and next human action.

Useful context:
- Main repo: $REPO
- Run worktrees usually live under: $HOME/agentflow-runs/worktrees
- Cron logs: $REPO/logs/orchestrator
- Run artifacts: $REPO/runs
PROMPT
)

  codex_args=(
    --ask-for-approval never
    exec
    --cd "$REPO"
    --add-dir "$HOME/agentflow-runs"
    --sandbox danger-full-access
    --output-last-message "$SUMMARY_LOG"
  )
  if [[ -n "$CODEX_MODEL" ]]; then
    codex_args+=(--model "$CODEX_MODEL")
  fi
  codex_args+=("$prompt")

  set +e
  "$CODEX_BIN" "${codex_args[@]}"
  status=$?
  set -e

  if (( status == 0 )); then
    if git status --short --untracked-files=normal | grep -q .; then
      echo "Recovery left main worktree dirty; treating as failed recovery"
      git status --short --untracked-files=normal
      status=80
    elif find "$HOME/agentflow-runs/worktrees" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null | grep -q .; then
      echo "Recovery left AgentFlow run worktrees behind; treating as failed recovery"
      find "$HOME/agentflow-runs/worktrees" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null
      status=81
    fi
  fi

  echo ""
  echo "Finished: $(date --iso-8601=seconds)"
  echo "Exit: $status"
} > "$RECOVERY_LOG" 2>&1

ln -sfn "$RECOVERY_LOG" "$LATEST_LOG"
echo "Codex recovery log: $RECOVERY_LOG"
if [[ -s "$SUMMARY_LOG" ]]; then
  echo ""
  echo "Codex recovery summary:"
  sed -n '1,120p' "$SUMMARY_LOG"
else
  echo "Codex recovery produced no summary. Last log lines:"
  tail -40 "$RECOVERY_LOG"
fi
exit "$status"
