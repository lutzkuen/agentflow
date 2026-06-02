#!/usr/bin/env bash
# Install the AgentFlow orchestrator as a systemd user timer (runs every 2 hours).
# Run once: bash scripts/install_timer.sh
# To remove: systemctl --user disable --now agentflow-orchestrator.timer

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"

# ── Proxy service unit ───────────────────────────────────────────────────────
cat > "$UNIT_DIR/agentflow-claude-proxy.service" << EOF
[Unit]
Description=AgentFlow Claude Proxy
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStartPre=-/usr/bin/fuser -k 4000/tcp
ExecStart=$REPO/.venv/bin/python -m uvicorn agentflow_proxy.server:app --host 127.0.0.1 --port 4000
Restart=always
RestartSec=1

[Install]
WantedBy=default.target
EOF

# ── Orchestrator service unit ────────────────────────────────────────────────
cat > "$UNIT_DIR/agentflow-orchestrator.service" << EOF
[Unit]
Description=AgentFlow Orchestrator
Wants=agentflow-claude-proxy.service
After=network.target agentflow-claude-proxy.service

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
systemctl --user enable --now agentflow-claude-proxy.service
systemctl --user enable --now agentflow-orchestrator.timer

echo ""
echo "Timer installed and started."
echo ""
echo "Proxy status:"
systemctl --user status agentflow-claude-proxy.service --no-pager
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
echo "  systemctl --user disable --now agentflow-claude-proxy.service"
echo "  rm $UNIT_DIR/agentflow-orchestrator.{service,timer} $UNIT_DIR/agentflow-claude-proxy.service"
