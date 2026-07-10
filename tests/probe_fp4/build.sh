#!/usr/bin/env bash
# Builds probe_fp4.cu against the *pixi-pinned* cudart/cuBLASLt (12.9.2.10 —
# the exact version audited in docs/ai/fp4_modular_support_research.md),
# using the system CUDA 13.0 toolkit's nvcc + headers (the pixi env ships
# no headers, only .so files; the header ABI for the enums/attrs used here
# — CUDA_R_4F_E2M1, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, the block-scale
# descriptor attributes — is stable across 12.8+/13.0).
#
# Deliberately does NOT use `pixi install`/`update` (the pixi env here is a
# read-only symlink into the main repo checkout; see docs/ai/memory note
# bf16-build-needs-gpu-only-dispatch / worktree conventions).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLIB="$REPO_ROOT/.pixi/envs/cuda/targets/sbsa-linux/lib"
PINC="/usr/local/cuda-13.0/targets/sbsa-linux/include"

CUBLASLT="$(ls "$PLIB"/libcublasLt.so.12.* | grep -v '\.so\.12$' | head -1)"
CUDART="$(ls "$PLIB"/libcudart.so.12.* | grep -v '\.so\.12$' | head -1)"

if [[ ! -f "$CUBLASLT" || ! -f "$CUDART" ]]; then
    echo "error: could not find pixi cudart/cublasLt under $PLIB" >&2
    exit 1
fi

echo "Building against:"
echo "  cublasLt: $CUBLASLT"
echo "  cudart:   $CUDART"
echo "  headers:  $PINC"

nvcc -std=c++17 -O2 -arch=sm_121 -I"$PINC" \
    "$SCRIPT_DIR/probe_fp4.cu" -o "$SCRIPT_DIR/probe_fp4" \
    -Xlinker "$CUBLASLT" -Xlinker "$CUDART" \
    -Xlinker -rpath="$PLIB"

echo "Built $SCRIPT_DIR/probe_fp4"
echo "Run with: flock -w 10800 /tmp/llmm-gpu.lock -c $SCRIPT_DIR/probe_fp4"
