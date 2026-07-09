#!/usr/bin/env bash
# GPT-2 124M from-scratch pretraining on FineWeb classic 10B (GPT-2 tokens).
# Mirrors the llm.c 124M reproduction hyperparameters, adapted to our flags.
#
# Prereqs:
#   - data/.fineweb10B/ fully written by `pixi run python data/fineweb.py -t classic -v 10B -m gpt-2`
#     (~103 shards: shard 000000 is val, the rest train)
#   - build/train_gpt2_bf16 built with: make build  # plus -D LLMM_BF16=1 variant
#
# NOTE: batch size. Swept 2026-07-04 on the idle GB10 (bf16, T=1024):
# B=4 26.5k tok/s, B=16 25.2k, B=32 29.0k, B=64 27.0k — bandwidth-bound, so
# B barely matters. B=32 (~38GB) is the default; grad accumulation keeps the
# effective batch at -d 524288 tokens regardless of B, so B only affects
# speed, not the training trajectory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/train_gpt2_bf16}"
B="${B:-32}"

if [ -z "${MOJO_PYTHON_LIBRARY:-}" ]; then
    for lib in .pixi/envs/default/lib/libpython3*.so .pixi/envs/default/lib/libpython3*.dylib; do
        [ -f "$lib" ] && export MOJO_PYTHON_LIBRARY="$lib" && break
    done
fi

# Sampling disabled (-s 0): the end-of-run generation step hit
# CUDA_ERROR_MISALIGNED_ADDRESS on 2026-07-09 (train_gpt2.mojo:1905, a B=1
# inference-shape bug never exercised before last_step) and crashed BEFORE
# the checkpoint write, losing the final model_19552.bin. -s 0 disables
# sampling unconditionally (last_step normally forces it regardless of
# sample_every) so the checkpoint write is reached this time.
exec "$BIN" \
    -i "data/.fineweb10B/fineweb_train_*.bin" \
    -j "data/.fineweb10B/fineweb_val_*.bin" \
    -e d12 \
    -o log124M \
    -n 1000 \
    -y 1 \
    -b "$B" \
    -t 1024 \
    -d 524288 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -c 0.1 \
    -v 250 \
    -s 0 \
    -x -1 \
    "$@"
