#!/usr/bin/env bash
# Idempotent: (re)launch the autosentry supervisor (per .autosentry/autosentry.yaml
# in this repo) if it isn't already running. No-ops if it's already up, so it's
# safe to call from anywhere — a user crontab `@reboot` entry (survives a machine
# reboot, no session/sudo needed) and a periodic liveness check alike.
#
# Cron runs with a minimal PATH, so this resolves the `autosentry` binary itself
# rather than assuming it's found the same way an interactive shell finds it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if pgrep -f "autosentr[y] run" >/dev/null; then
	exit 0
fi

AUTOSENTRY="$(command -v autosentry || true)"
if [ -z "$AUTOSENTRY" ]; then
	for candidate in "$HOME/.local/bin/autosentry" /usr/local/bin/autosentry; do
		if [ -x "$candidate" ]; then
			AUTOSENTRY="$candidate"
			break
		fi
	done
fi
if [ -z "$AUTOSENTRY" ]; then
	echo "error: autosentry not found on PATH or in common install locations" >&2
	exit 1
fi

mkdir -p .autosentry
setsid nohup "$AUTOSENTRY" run >.autosentry/nohup.out 2>&1 </dev/null &
disown
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ensure_supervisor: launched autosentry (was not running)" >>.autosentry/ensure_supervisor.log
