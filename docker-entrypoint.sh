#!/usr/bin/env sh
set -eu

mkdir -p "${TOKENCLAW_CONFIG_DIR:-/config}"

db="${TOKENCLAW_DB:-/data/tokenclaw.sqlite3}"
case "$db" in
  sqlite:///*)
    db_path="${db#sqlite:///}"
    ;;
  *://*)
    db_path=""
    ;;
  *)
    db_path="$db"
    ;;
esac

if [ -n "$db_path" ]; then
  mkdir -p "$(dirname "$db_path")"
fi

if [ "$#" -eq 0 ]; then
  set -- tokenclaw start
fi

case "$1" in
  -h|--help|--version|start|run|activate|deactivate|stats|doctor|internal|demo|db|savings|version)
    set -- tokenclaw "$@"
    ;;
esac

exec "$@"
