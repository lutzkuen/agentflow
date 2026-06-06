#!/usr/bin/env bash
# Install hourly AgentFlow orchestrator cron retry.
# The orchestrator itself records a cooldown after worker rate limits, so hourly
# cron remains a health heartbeat without necessarily spending a model call.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$REPO/scripts/run_orchestrator_cron.sh"
ARCHIVER="$REPO/scripts/archive_orchestrator_logs.py"
CRON_MINUTE=${AGENTFLOW_CRON_MINUTE:-17}
ARCHIVE_CRON_MINUTE=${AGENTFLOW_ARCHIVE_CRON_MINUTE:-7}
ARCHIVE_CRON_HOUR=${AGENTFLOW_ARCHIVE_CRON_HOUR:-2}
START_MARK="# BEGIN AgentFlow hourly orchestrator"
END_MARK="# END AgentFlow hourly orchestrator"

chmod +x "$WRAPPER"
chmod +x "$ARCHIVER"
mkdir -p "$REPO/logs/orchestrator" "$REPO/runs"

tmp_current=$(mktemp)
tmp_next=$(mktemp)
trap 'rm -f "$tmp_current" "$tmp_next"' EXIT

crontab -l > "$tmp_current" 2>/dev/null || true

awk -v start="$START_MARK" -v end="$END_MARK" '
  $0 == start { skip = 1; next }
  $0 == end { skip = 0; next }
  !skip { print }
' "$tmp_current" > "$tmp_next"

{
  cat "$tmp_next"
  if [[ -s "$tmp_next" ]] && [[ "$(tail -c 1 "$tmp_next")" != "" ]]; then
    echo ""
  fi
  echo "$START_MARK"
  echo "SHELL=/bin/bash"
  echo "PATH=/home/$USER/.local/bin:/usr/local/bin:/usr/bin:/bin"
  echo "$CRON_MINUTE * * * * $WRAPPER"
  echo "$ARCHIVE_CRON_MINUTE $ARCHIVE_CRON_HOUR * * * $ARCHIVER --log-dir $REPO/logs/orchestrator >> $REPO/logs/orchestrator/archive.log 2>&1"
  echo "$END_MARK"
} | crontab -

echo "Installed hourly AgentFlow cron retry:"
echo "  $CRON_MINUTE * * * * $WRAPPER"
echo "  Worker defaults to Codex; set AGENTFLOW_WORKER=claude in $REPO/.env to switch back."
echo "  Worker rate-limit cooldown defaults to 90 minutes between upstream retry attempts."
echo ""
echo "Installed nightly AgentFlow orchestrator log archive:"
echo "  $ARCHIVE_CRON_MINUTE $ARCHIVE_CRON_HOUR * * * $ARCHIVER --log-dir $REPO/logs/orchestrator"
echo "  Quota-only logs older than 24 hours are deleted; work logs are archived under YYYY/MM/DD."
echo ""
echo "Cron stdout/stderr logs:"
echo "  $REPO/logs/orchestrator/"
echo ""
echo "Orchestrator run summaries:"
echo "  $REPO/runs/"
