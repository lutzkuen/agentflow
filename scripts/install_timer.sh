#!/usr/bin/env bash
# Install the AgentFlow orchestrator as a systemd user timer (runs every 2 hours).
# Run once: bash scripts/install_timer.sh
# To remove: systemctl --user disable --now agentflow-orchestrator.timer

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"

# ── Service unit ─────────────────────────────────────────────────────────────
cat > "$UNIT_DIR/agentflow-orchestrator.service" << EOF
[Unit]
Description=AgentFlow Orchestrator
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$REPO
Environment=ANTHROPIC_BASE_URL=http://127.0.0.1:4000
ExecStart=/usr/bin/env bash $REPO/agents/orchestrate.sh
StandardOutput=journal
StandardError=journal
# Keep PATH that includes claude CLI
Environment=PATH=/home/$USER/.local/bin:/usr/local/bin:/usr/bin:/bin
EOF

# ── Timer unit ───────────────────────────────────────────────────────────────
cat > "$UNIT_DIR/agentflow-orchestrator.timer" << EOF
[Unit]
Description=AgentFlow Orchestrator — every 2 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=2h
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now agentflow-orchestrator.timer

echo ""
echo "Timer installed and started."
echo ""
echo "Status:"
systemctl --user status agentflow-orchestrator.timer --no-pager
echo ""
echo "To see next run time:"
echo "  systemctl --user list-timers agentflow-orchestrator.timer"
echo ""
echo "To run immediately:"
echo "  systemctl --user start agentflow-orchestrator.service"
echo ""
echo "To follow logs:"
echo "  journalctl --user -u agentflow-orchestrator.service -f"
echo ""
echo "To uninstall:"
echo "  systemctl --user disable --now agentflow-orchestrator.timer"
echo "  rm $UNIT_DIR/agentflow-orchestrator.{service,timer}"
