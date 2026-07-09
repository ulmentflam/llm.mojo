#!/usr/bin/env bash
# Resume the FineWeb classic-10B download/tokenization after a reboot.
# HuggingFace hub caches completed parquet files, so this picks up roughly
# where the killed run left off. Safe to re-run; exits early once all
# ~103 shards exist in data/.fineweb10B/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

n=$(ls data/.fineweb10B/ 2>/dev/null | wc -l)
if [ "$n" -ge 103 ]; then
    echo "data/.fineweb10B already has $n shards — nothing to do."
    exit 0
fi

echo "Shards present: $n/~103 — starting download/tokenize (log: scratch/fineweb_download.log)"
nohup pixi run python data/fineweb.py -t classic -v 10B -m gpt-2 \
    > scratch/fineweb_download.log 2>&1 &
echo "PID: $!"
