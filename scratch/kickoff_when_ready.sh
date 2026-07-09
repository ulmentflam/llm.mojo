#!/usr/bin/env bash
# Wait for the fineweb tokenization to finish, verify the shard set, then
# launch the 124M FineWeb training run detached (survives terminal/session).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DIR=data/.fineweb10B

# 1. Wait for the tokenizer process to exit.
while pgrep -f "python data/fineweb.py" >/dev/null; do sleep 15; done

# 2. Verify: 103 shards, val + train_000001..000102, full size except the last.
n=$(ls "$DIR" | wc -l)
if [ "$n" -lt 103 ]; then
    echo "ABORT: tokenizer exited with only $n/103 shards — not launching training."
    exit 1
fi
bad=0
for f in "$DIR"/fineweb_val_000000.bin $(ls "$DIR"/fineweb_train_*.bin | head -n -1); do
    sz=$(stat -c%s "$f")
    [ "$sz" -eq 200001024 ] || { echo "BAD SIZE: $f = $sz"; bad=1; }
done
# every shard must carry the gpt-2 magic/version header
for f in "$DIR"/*.bin; do
    read -r magic version < <(od -A n -t d4 -N 8 "$f" | awk '{print $1, $2}')
    [ "$magic" = "20240520" ] && [ "$version" = "1" ] || { echo "BAD HEADER: $f ($magic/$version)"; bad=1; }
done
[ "$bad" -eq 0 ] || { echo "ABORT: shard verification failed."; exit 1; }
echo "Verified: $n shards, sizes and headers OK."

# 3. Launch training under the autosentry supervisor (self-healing: restarts
#    with checkpoint resume, OOM batch-halving, Claude escalation).
mkdir -p log124M
setsid nohup autosentry run > scratch/autosentry.out 2>&1 &
APID=$!
sleep 10
if kill -0 "$APID" 2>/dev/null && pgrep -f "build/train_gpt2" >/dev/null; then
    echo "autosentry supervising training: supervisor PID $APID, trainer up."
    echo "Logs: .autosentry/logs/  State: .autosentry/state.json  Checkpoints: log124M/"
else
    echo "ABORT: supervisor or trainer failed to start:"
    tail -15 scratch/autosentry.out; tail -10 .autosentry/logs/* 2>/dev/null | tail -15
    exit 1
fi
