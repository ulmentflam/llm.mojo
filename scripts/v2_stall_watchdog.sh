#!/usr/bin/env bash
# The external log-mtime watchdog that .autosentry/autosentry.yaml has always
# claimed exists.
#
# That config omits a `stall` detector on purpose -- the built-in one
# kill-looped the trainer during the 2026-07-05 GB10 run by tracking a metric
# that did not re-attach to a restarted child -- and says "Hang coverage comes
# from an external log-mtime watchdog instead." No such watchdog was ever
# written. The comment reads as though coverage lives elsewhere, so nobody goes
# looking, which is worse than admitting there is none.
#
# It is needed because autosentry can wedge while looking perfectly healthy:
# on 2026-07-28 the supervisor sat in futex_wait_queue with zero children,
# heartbeating every 30s, `stage: None`, for THIRTEEN HOURS between two arms.
# `pgrep autosentry` was true the entire time. Process liveness is not work
# liveness -- so this watches the work.
#
# Deliberately not a Claude session monitor: those die with the session, and
# both 13-hour and 14-hour outages here happened while no session was watching.
# Cron survives logout, reboot and agent teardown.
#
# Install:
#   */15 * * * * /home/evan/workspace/llm.mojo/scripts/v2_stall_watchdog.sh
set -uo pipefail

# Overridable so the alert path is testable. A watchdog whose firing path has
# never been executed is a guard nobody has seen work -- the exact shape of
# defect this repo spent a week removing from its test suite.
RUNS=${RUNS:-/data/llm.mojo/runs/v2}
RESULTS=${RESULTS:-/home/evan/workspace/llm.mojo/docs/ai/v2_arm_results}
STALL_SECS=${STALL_SECS:-1800}
LOG=${LOG:-/home/evan/workspace/llm.mojo/.autosentry/stall_watchdog.log}
STATE=${STATE:-/tmp/.llmm_stall_watchdog_last}
DRY_RUN=${DRY_RUN:-0}

# Nothing to supervise: no run dirs, or every arm already published.
shopt -s nullglob
dirs=("$RUNS"/*_v2)
((${#dirs[@]})) || exit 0

published=0
for d in "${dirs[@]}"; do
	arm=$(basename "$d" | sed -E 's/^log([0-9]+M)_fineweb_(.*)_v2$/\1-\2/')
	[ -f "$RESULTS/$arm.json" ] && published=$((published + 1))
done
[ "$published" -ge 6 ] && exit 0

newest=0
for f in "$RUNS"/*_v2/train.log "$RUNS"/*_v2/publish.log; do
	[ -f "$f" ] || continue
	m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
	[ "$m" -gt "$newest" ] && newest=$m
done
[ "$newest" -eq 0 ] && exit 0

quiet=$(($(date +%s) - newest))
[ "$quiet" -le "$STALL_SECS" ] && { rm -f "$STATE"; exit 0; }

# Training appends every step (sub-second at both scales); the HellaSwag eval
# inside the publish step appends continuously for ~10 min. Only the Hub upload
# is briefly quiet. Silence everywhere past the threshold is a stall.
sup=$(pgrep -f "autosentr[y] run" >/dev/null && echo "alive" || echo "DEAD")
trainers=$(pgrep -cf "build/train_gpt2" 2>/dev/null || echo 0)
msg="$(date -u +%Y-%m-%dT%H:%M:%SZ) STALL: no v2 log written for $((quiet / 60)) min"
msg="$msg (supervisor=$sup trainers=$trainers published=$published/6)"

# Report once per stall episode, not every 15 minutes -- an alert that repeats
# is one you learn to filter.
[ "$(cat "$STATE" 2>/dev/null || echo)" = "stalled" ] && exit 0
echo "stalled" >"$STATE"
echo "$msg" >>"$LOG"
command -v wall >/dev/null && echo "$msg" | wall 2>/dev/null

if [ "$DRY_RUN" = "1" ]; then
	exit 0
elif [ "$sup" = "DEAD" ]; then
	# A dead supervisor is the recoverable case and ensure_supervisor is
	# idempotent, so fix it rather than only reporting it.
	/home/evan/workspace/llm.mojo/scripts/ensure_supervisor.sh &&
		echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) relaunched supervisor" >>"$LOG"
else
	# Alive but not advancing is the wedge, and recovering it means deciding
	# which stages to drop so completed arms are not re-run. That is a
	# judgement call, so surface it instead of guessing.
	echo "  supervisor alive but not advancing -- the known wedge." >>"$LOG"
	echo "  kill -9 it, park .autosentry/state.json, drop completed stages, relaunch." >>"$LOG"
fi
