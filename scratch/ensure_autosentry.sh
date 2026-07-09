#!/usr/bin/env bash
# Idempotent: (re)launch the autosentry supervisor for the FineWeb 124M
# training run if it isn't already running. Meant to be called from a user
# crontab @reboot entry (survives machine reboot, no session/sudo needed) and
# safe to call from anywhere (e.g. a periodic cron liveness check) since it
# no-ops when autosentry is already up.
set -euo pipefail
ROOT="/home/evan/workspace/llm.mojo"
cd "$ROOT"

if pgrep -f "autosentr[y] run" >/dev/null; then
    exit 0
fi

mkdir -p log124M
setsid nohup autosentry run > scratch/autosentry.out 2>&1 < /dev/null &
disown
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ensure_autosentry: launched supervisor (was not running)" >> scratch/ensure_autosentry.log
