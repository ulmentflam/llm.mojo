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

# Give the supervisor -- and therefore every stage script it launches -- a PATH
# that contains the tools those scripts call. Resolving `autosentry` above is
# not enough: cron's PATH is
# /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:...
# so a supervisor started by the */10 liveness entry hands its children a PATH
# with no `pixi` in it, and anything calling `pixi run` dies with exit 127.
#
# That is not hypothetical. On 2026-07-29 cron relaunched the supervisor at
# 12:09; the publish step of a completed training arm then failed 55 times over
# ~14 hours, each retry re-running a 10-minute HellaSwag eval on a finished
# checkpoint. Commit cb331e4 fixed exactly this for `autosentry` itself and
# stopped one level short of the scripts it starts.
for d in "$HOME/.pixi/bin" "$HOME/.local/bin" /usr/local/bin; do
	case ":$PATH:" in
	*":$d:"*) ;;
	*) [ -d "$d" ] && PATH="$d:$PATH" ;;
	esac
done
export PATH

mkdir -p .autosentry
setsid nohup "$AUTOSENTRY" run >.autosentry/nohup.out 2>&1 </dev/null &
disown
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ensure_supervisor: launched autosentry (was not running)" >>.autosentry/ensure_supervisor.log
