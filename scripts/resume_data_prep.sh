#!/usr/bin/env bash
# Resume a possibly-interrupted data-prep job (e.g. data/fineweb.py,
# data/hellaswag.py) idempotently: no-ops if OUTPUT_DIR already has at least
# MIN_COUNT files, otherwise (re)starts PREP_CMD detached. Safe to re-run —
# most data-prep scripts in this repo (and HuggingFace's own dataset/hub
# caching) skip work that's already done, so this just picks up roughly where
# a killed run left off rather than restarting from scratch.
#
# Configure via environment variables:
#   OUTPUT_DIR  directory the prep job writes into (required)
#   MIN_COUNT   file count in OUTPUT_DIR that means "done" (required)
#   PREP_CMD    command to (re)start the prep job (required)
#   LOG_FILE    where to redirect PREP_CMD's output (default: derived from
#               OUTPUT_DIR's basename, under build/)
#
# Example:
#   OUTPUT_DIR=data/.fineweb10B MIN_COUNT=103 \
#   PREP_CMD="pixi run python data/fineweb.py -t classic -v 10B -m gpt-2" \
#   ./scripts/resume_data_prep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${MIN_COUNT:?MIN_COUNT is required}"
: "${PREP_CMD:?PREP_CMD is required}"
LOG_FILE="${LOG_FILE:-build/$(basename "$OUTPUT_DIR" | sed 's/^\.//')_prep.log}"

n=$(ls "$OUTPUT_DIR" 2>/dev/null | wc -l)
if [ "$n" -ge "$MIN_COUNT" ]; then
	echo "$OUTPUT_DIR already has $n files (>= $MIN_COUNT) — nothing to do."
	exit 0
fi

mkdir -p "$(dirname "$LOG_FILE")"
echo "Files present: $n/$MIN_COUNT — starting prep (log: $LOG_FILE)"
nohup bash -c "$PREP_CMD" >"$LOG_FILE" 2>&1 &
echo "PID: $!"
