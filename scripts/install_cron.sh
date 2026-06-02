#!/usr/bin/env bash
# Install hourly AgentFlow orchestrator cron retry.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$REPO/scripts/run_orchestrator_cron.sh"
CRON_MINUTE=${AGENTFLOW_CRON_MINUTE:-17}
START_MARK="# BEGIN AgentFlow hourly orchestrator"
END_MARK="# END AgentFlow hourly orchestrator"

chmod +x "$WRAPPER"
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
  echo "$END_MARK"
} | crontab -

echo "Installed hourly AgentFlow cron retry:"
echo "  $CRON_MINUTE * * * * $WRAPPER"
echo ""
echo "Cron stdout/stderr logs:"
echo "  $REPO/logs/orchestrator/"
echo ""
echo "Orchestrator run summaries:"
echo "  $REPO/runs/"
