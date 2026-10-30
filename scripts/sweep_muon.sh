#!/usr/bin/env bash
# Run ONE tagged config of the Muon hyperparameter sweep (300-step d12 ranking run).
#
#   scripts/sweep_muon.sh <tag> <varying trainer flags...>
#   e.g. scripts/sweep_muon.sh lr6e-4 -opt muon -l 6e-4 -mu 0.95 -c 0.1
#
# Fixed recipe is below (override -gc etc. by passing the flag again in the
# varying part; later flags win in the trainer's parser, check the setup line).
# Writes sweeps/muon_<tag>.jsonl (LLMM_LOG_FILE, verbose) and the console log to
# $SWEEP_JOBS/<tag>.log (default: /tmp/llmm_sweep). Refuses to start if another
# train_gpt2 process is running (runs must be sequential). Run it in the
# background; summarise with: python3 scripts/compare_runs.py --table sweeps/muon_*.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n 2,12p "$0"; exit 2; }
tag=$1; shift
jobs_dir=${SWEEP_JOBS:-/tmp/llmm_sweep}
mkdir -p "$jobs_dir" sweeps
# Only a trainer that holds the GPU blocks us (sibling CPU-target jobs never
# appear in nvidia-smi). Wait up to ~15 min for a GPU trainer to exit.
gpu_busy() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q train_gpt2; }
for i in $(seq 15); do
  gpu_busy || break
  [ "$i" = 15 ] && { echo "refusing: a train_gpt2 process is using the GPU" >&2; exit 1; }
  echo "GPU trainer running, retry $i/15 in 60s"; sleep 60
done
bin=build/train_gpt2_bf16
ls -l --time-style=full-iso "$bin" | sed 's/^/binary: /'
log=sweeps/muon_${tag}.jsonl
rm -f "$log"
common="-e d12 -i 'data/.fineweb10B/fineweb_train_*.bin' -j 'data/.fineweb10B/fineweb_val_*.bin' -b 32 -t 1024 -d 131072 -u 50 -q 0.0 -gc 1.0 -x ${SWEEP_STEPS:-300} -v ${SWEEP_VAL:-100} -m 20 -s 0 -qk_clip 0"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee "$jobs_dir/${tag}.gpu_before"
LLMM_LOG_LEVEL=verbose LLMM_LOG_FILE="$log" make train-bf16 ARGS="$common $*" >"$jobs_dir/${tag}.log" 2>&1
rc=$?
echo "exit=$rc"; tail -n 2 "$jobs_dir/${tag}.log"
exit $rc
