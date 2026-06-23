#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${AGENTFLOW_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -n "${AGENTFLOW_TARGET_PYTHON:-}" ]]; then
  PYTHON_BIN="${AGENTFLOW_TARGET_PYTHON}"
elif [[ -n "${AGENTFLOW_VENV:-}" && -x "${AGENTFLOW_VENV}/bin/python" ]]; then
  PYTHON_BIN="${AGENTFLOW_VENV}/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

TOKENCLAW_DEV_DB="${TOKENCLAW_DEV_DB:-${HOME}/.tokenclaw/dev.sqlite3}"
mkdir -p "$(dirname "${TOKENCLAW_DEV_DB}")"

export TOKENCLAW_PROVIDER="anthropic"
export TOKENCLAW_HOST="127.0.0.1"
export TOKENCLAW_PORT="4001"
export TOKENCLAW_DB="${TOKENCLAW_DEV_DB}"
unset TOKENCLAW_DATABASE_URL

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m tokenclaw.server \
  --provider anthropic \
  --host "${TOKENCLAW_HOST}" \
  --port "${TOKENCLAW_PORT}"
