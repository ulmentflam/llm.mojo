#!/usr/bin/env bash
# Wait for a data-prep job (e.g. data/fineweb.py, data/hellaswag.py) to finish,
# verify the resulting llm.mojo/llm.c-format .bin shards, then run a launch
# command (typically scripts/ensure_supervisor.sh, or a training/eval command
# directly) — detached, so it survives the calling terminal/session exiting.
#
# Configure via environment variables (all but SHARD_DIR/MIN_SHARDS/LAUNCH_CMD
# have sensible defaults for this repo's own data format):
#   WAIT_PROC_PATTERN  pgrep -f pattern to wait on before verifying (optional;
#                      skip the wait entirely if unset, e.g. prep already ran)
#   SHARD_DIR          directory containing the shards (required)
#   SHARD_GLOB         glob within SHARD_DIR to count/verify (default "*.bin")
#   MIN_SHARDS         minimum shard count to consider prep complete (required)
#   FULL_SHARD_BYTES   if set, every shard except the lexicographically last
#                      one must be exactly this size (catches a truncated
#                      mid-write shard from a killed prep job)
#   MAGIC, VERSION     header fields every shard must carry (default: this
#                      repo's own format, see llmm/checkpointing.mojo)
#   LAUNCH_CMD         command to exec once verified (required)
#
# Example:
#   WAIT_PROC_PATTERN="python data/fineweb.py" SHARD_DIR=data/.fineweb10B \
#   MIN_SHARDS=103 FULL_SHARD_BYTES=200001024 \
#   LAUNCH_CMD="scripts/ensure_supervisor.sh" \
#   ./scripts/wait_for_shards_then_launch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHARD_GLOB="${SHARD_GLOB:-*.bin}"
MAGIC="${MAGIC:-20240520}"
VERSION="${VERSION:-1}"

: "${SHARD_DIR:?SHARD_DIR is required}"
: "${MIN_SHARDS:?MIN_SHARDS is required}"
: "${LAUNCH_CMD:?LAUNCH_CMD is required}"

# 1. Wait for the prep job to exit, if one's still running.
if [ -n "${WAIT_PROC_PATTERN:-}" ]; then
	while pgrep -f "$WAIT_PROC_PATTERN" >/dev/null; do sleep 15; done
fi

# 2. Verify: enough shards, right sizes, right header.
n=$(ls "$SHARD_DIR"/$SHARD_GLOB 2>/dev/null | wc -l)
if [ "$n" -lt "$MIN_SHARDS" ]; then
	echo "ABORT: only $n/$MIN_SHARDS shards in $SHARD_DIR — not launching." >&2
	exit 1
fi
bad=0
if [ -n "${FULL_SHARD_BYTES:-}" ]; then
	# Any dataset that doesn't divide evenly into fixed-size shards has
	# exactly one natural partial "remainder" shard — which one that is
	# depends on the prep script's own naming/ordering, not something this
	# generic script should guess at. So: more than one undersized shard means
	# a truncated/interrupted write (a real failure); at most one is normal.
	undersized=0
	for f in "$SHARD_DIR"/$SHARD_GLOB; do
		sz=$(stat -c%s "$f")
		if [ "$sz" -ne "$FULL_SHARD_BYTES" ]; then
			undersized=$((undersized + 1))
			echo "note: $f = $sz bytes (expected $FULL_SHARD_BYTES) — ok if this is the one final/partial shard"
		fi
	done
	if [ "$undersized" -gt 1 ]; then
		echo "BAD SIZES: $undersized shards deviate from $FULL_SHARD_BYTES bytes (at most 1 expected — a truncated write?)" >&2
		bad=1
	fi
fi
for f in "$SHARD_DIR"/$SHARD_GLOB; do
	read -r magic version < <(od -A n -t d4 -N 8 "$f" | awk '{print $1, $2}')
	[ "$magic" = "$MAGIC" ] && [ "$version" = "$VERSION" ] || {
		echo "BAD HEADER: $f (magic=$magic version=$version, expected $MAGIC/$VERSION)" >&2
		bad=1
	}
done
[ "$bad" -eq 0 ] || {
	echo "ABORT: shard verification failed." >&2
	exit 1
}
echo "Verified: $n shards in $SHARD_DIR, sizes and headers OK."

# 3. Launch, detached — survives the calling session ending.
setsid nohup bash -c "$LAUNCH_CMD" >/dev/null 2>&1 &
disown
echo "Launched: $LAUNCH_CMD (PID $!)"
