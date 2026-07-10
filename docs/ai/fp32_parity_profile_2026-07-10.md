# fp32 parity profile — where llm.mojo fp32 loses ~120 ms/step vs llm.c fp32

**Date:** 2026-07-10. **Branch:** `goal1/fp32-parity`. **Machine:** NVIDIA GB10
(Grace-Blackwell, aarch64, unified memory), DGX Spark, 20 cores, driver
595.71.05, CUDA 13.2, ncu 2025.3.1.

Goal: a data-backed breakdown of the llm.mojo-fp32 vs llm.c-fp32 gap
(417.72 vs 298.53 ms/step in the 2026-07-03 official benchmark), as the
pre-fix baseline for the fp32-parity campaign. A parallel static analysis
already identified the primary suspect — llm.mojo hardcodes
`ComputeType.COMPUTE_32F` (TF32 disabled) in `llmm/matmul.mojo:214` and
`llmm/attention.mojo:4303`, while llm.c fp32 auto-enables
`CUBLAS_TF32_TENSOR_OP_MATH` on sm≥80 (`train_gpt2_fp32.cu:1614-1617`).
This profile confirms the gap is concentrated exactly where that predicts,
and quantifies the per-kernel distribution.

## Measurement conditions

- GPU exclusively held via `flock /tmp/llmm-gpu.lock` for every run; both
  arms interleaved (A B A B) inside one quiet window.
- Pre-run box state: `free -g` = 121 total / 74 used / 47 available GB;
  `nvidia-smi`: 0% util, 40 °C, only Xorg/gnome-shell resident (~25 MiB).
  No vLLM tenant (the 87 GB allocation-footprint slowdown documented in
  memory does not apply to this window).
- Binaries freshly built this session (mtimes 12:49–12:50, sources 12:45):
  `build/profile_gpt2` (fp32, `make build-profile` recipe),
  `third_party/llm.c/train_gpt2cu`, `third_party/llm.c/train_gpt2fp32cu`
  (`nvcc -O3 --use_fast_math`, NO_MULTI_GPU=1, CUDA-13 cudaMemAdvise patch).

## A/B benchmark (B=4, T=1024, L=12, interleaved, 2 reps each)

Harness: same code paths as `scripts/benchmark_train.py` (its `bench_mojo` /
`bench_llmc_cuda_fp32` helpers driven by a small fp32-only driver), 25 steps
per llm.mojo rep with the harness's 5-step warmup trim → n=20/rep. The llm.c
fp32 binary has no `-x` flag and runs one epoch of the val bin = 8 steps
→ n=3/rep after trim (same limitation as the 2026-07-03 official run, which
also had n=3 for this arm).

| arm | rep | n | mean ms | median ms | std | tok/s |
|---|---|---:|---:|---:|---:|---:|
| llm.mojo fp32 | 1 | 20 | 414.97 | 414.53 | 1.59 | 9,871 |
| llm.c fp32 | 1 | 3 | 291.12 | 291.16 | 0.17 | 14,070 |
| llm.mojo fp32 | 2 | 20 | 418.34 | 418.04 | 1.70 | 9,791 |
| llm.c fp32 | 2 | 3 | 296.90 | 296.57 | 1.63 | 13,796 |
| **llm.mojo pooled** | — | 40 | **416.66** | — | — | **9,831** |
| **llm.c pooled** | — | 6 | **294.01** | — | — | **13,931** |

**Gap: 122.6 ms/step, ratio 1.417×** — matches the 2026-07-03 baseline
(417.72 vs 298.53, 1.40×) within noise. tok/s sanity: 4·1024/416.66 ms
= 9,831 ✓; 4096/294.01 = 13,931 ✓.

llm.mojo per-phase (pooled across reps, n=40):

| phase | mean ms | share |
|---|---:|---:|
| forward | 155.62 | 37.3% |
| backward | 244.05 | 58.6% |
| update | 16.99 | 4.1% |
| **total** | **416.66** | |

(bf16 reference for scale, 2026-07-03: fwd ~50 / bwd ~83 / upd ~16.5 —
fp32's update matches bf16's; the entire fp32 penalty is fwd+bwd.)

## llm.mojo fp32 per-kernel profile (ncu, B=4 T=1024 L=12, 1 step)

`make profile-fp32-ncu`-equivalent invocation (`profile_gpt2.py --exe
build/profile_gpt2`, PROFILE_T=1024), under the GPU lock. Total captured GPU
time 435.8 ms (ncu-serialized; wall step is 417 ms). Perf-counter access
unavailable without sudo, so DRAM/tensor% columns are absent — but the
kernel *names* are decisive (see below). Top 15 of 33 kernels:

| kernel | family | calls | total ms | % of step |
|---|---|---:|---:|---:|
| magma_sgemmEx_kernel<float,float,float,…> | matmul | 72 | 89.50 | 20.5 |
| cutlass_80_**simt**_sgemm_64x6… | matmul | 24 | 52.66 | 12.1 |
| cutlass_80_**simt**_sgemm_128x… | matmul | 37 | 48.71 | 11.2 |
| cutlass_80_**simt**_sgemm_128x… | matmul | 13 | 37.30 | 8.6 |
| cutlass_80_**simt**_sgemm_256x… | matmul | 36 | 30.32 | 7.0 |
| cutlass_80_**simt**_sgemm_128x… | matmul | 1 | 26.47 | 6.1 |
| cutlass_80_**simt**_sgemm_64x6… | matmul | 24 | 22.69 | 5.2 |
| llmm_attention_attention_bwd | attention | 12 | 16.06 | 3.7 |
| llmm_adamw_adamw_update_gpu | optimizer | 1 | 14.83 | 3.4 |
| cublasLt::globalKernel<8,32,float,…> | other | 12 | 14.74 | 3.4 |
| llmm_attention_attention_soft | attention | 12 | 11.32 | 2.6 |
| cutlass_80_**simt**_sgemm_128x… | matmul | 12 | 10.85 | 2.5 |
| llmm_fused_classifier | classifier | 1 | 8.25 | 1.9 |
| cublasLt::globalKernel<8,32,float,…> | other | 12 | 8.09 | 1.9 |
| llmm_layernorm_layernorm_bwd_r | layernorm | 24 | 7.45 | 1.7 |

By family (of 435.8 ms ncu total):

| family | calls | ms | % |
|---|---:|---:|---:|
| matmul | 315 | 325.56 | 74.7 |
| attention (non-GEMM) | 36 | 30.13 | 6.9 |
| other (cublasLt epilogue/split-K) | 24 | 22.84 | 5.2 |
| layernorm | 124 | 21.05 | 4.8 |
| optimizer | 3 | 16.82 | 3.9 |
| split/merge | 48 | 10.70 | 2.5 |
| classifier | 1 | 8.25 | 1.9 |
| encoder | 3 | 0.45 | 0.1 |

**The decisive observation: every one of the 219 GEMM launches is a
non-tensor-core kernel** — `cutlass_80_simt_sgemm_*` (CUDA-core SIMT FFMA)
plus `magma_sgemmEx_kernel` (cuBLASLt's MAGMA-derived sgemm on GB10). Zero
`tensorop` kernels anywhere in the capture. This is the direct kernel-level
signature of `COMPUTE_32F` (TF32 off): cuBLASLt is forbidden from using
tensor cores for these GEMMs.

## llm.c fp32 comparison profile (ncu)

Two captures of `train_gpt2fp32cu` at B=4 T=1024 under the lock:

1. **First 400 launches** (accidentally = step-0 validation forwards; kept
   because it cleanly isolates llm.c's *forward* GEMM behavior — one val
   forward = 172 launches):
   - `matmul_forward_kernel4` (llm.c's *hand-written SIMT* forward matmul —
     no cuBLAS, no TF32; `train_gpt2_fp32.cu:617,734`): 114 calls, 178.2 ms
     ncu-time, 56.1% of the window.
   - `cutlass_80_tensorop_s1688gemm_*` (TF32 **tensor-core** GEMMs, from the
     cuBLAS strided-batched attention matmuls): 2×28 calls, 62.5 ms, 19.6%.
   - So even llm.c fp32's *forward* dense GEMMs don't use tensor cores —
     only its attention (fwd+bwd) and dense *backward* GEMMs go through
     cuBLAS and get TF32.
2. **One full training step** (launch-skip 1376 = past the 8 val batches,
   trimmed at the adamw kernel at launch offset 440 → 441 launches, 307.7 ms
   ncu-serialized vs 294 ms wall). Top kernels:

| kernel | calls | total ms | % of step |
|---|---:|---:|---:|
| matmul_forward_kernel4 (hand-written SIMT) | 49 | 79.00 | 25.7 |
| cutlass_80_**tensorop**_s1688gemm_64x64_16x6 | 24 | 27.36 | 8.9 |
| cutlass_80_**tensorop**_s1688gemm_256x64_16x4 | 24 | 25.99 | 8.4 |
| cutlass_80_**tensorop**_s1688gemm_64x64_16x6 (2nd cfg) | 24 | 22.90 | 7.4 |
| cutlass_80_**tensorop**_s1688gemm_256x128_16x3 | 48 | 22.28 | 7.2 |
| softmax_autoregressive_backward_kernel | 12 | 16.28 | 5.3 |
| cutlass_80_**tensorop**_s1688gemm_128x128_32x3 | 36 | 15.59 | 5.1 |
| adamw_kernel2 | 1 | 14.69 | 4.8 |
| softmax_forward_kernel5 | 12 | 10.82 | 3.5 |
| fused_classifier_kernel3 | 1 | 9.35 | 3.0 |
| matmul_backward_bias_kernel4 | 48 | 8.91 | 2.9 |
| (3 more tensorop configs) | 14 | 20.84 | 6.8 |
| gelu_backward_kernel | 12 | 7.46 | 2.4 |
| layernorm_backward_kernel2 | 25 | 5.59 | 1.8 |
| gelu_forward_kernel | 12 | 5.15 | 1.7 |
| (permute/unpermute/residual/ln-fwd/encoder) | 99 | 15.55 | 5.1 |

llm.c per-phase, split at the launch boundaries (fused_classifier at 171,
adamw at 440), ncu-serialized: **fwd 141.11 / bwd 151.94 / upd 14.69 ms**.
GEMM-family within those: fwd 105.32, bwd 117.54 (total 222.86 ms, 72.4%).
Tensorop (TF32) kernels total 134.95 ms; the hand-written SIMT forward
matmul is 79.00 ms.

## Side-by-side: the gap is GEMMs, full stop

Family totals, both ncu-serialized captures of one step:

| family | llm.mojo ms | llm.c ms | delta |
|---|---:|---:|---:|
| **GEMM (incl. cuBLASLt epilogue/bias kernels)** | **348.40** | **222.86** | **+125.5** |
| attention non-GEMM (softmax fwd/bwd + permutes) | 40.83 | 36.97 | +3.9 |
| layernorm | 21.05 | 7.83 | +13.2 |
| gelu | 0 (fused in GEMM epilogue) | 12.61 | -12.6 |
| optimizer | 16.82 | 14.69 | +2.1 |
| classifier | 8.25 | 9.35 | -1.1 |
| residual | 0 (fused) | 3.24 | -3.2 |
| encoder | 0.45 | 0.19 | +0.3 |
| **total** | **435.79** | **307.74** | **+128.0** |

Non-GEMM totals: llm.mojo 87.4 vs llm.c 84.9 ms — **already at parity**
(the LN/gelu/residual differences are fusion bookkeeping, not real deltas).
The entire step gap sits in the GEMM row: +125.5 ms ncu-serialized, matching
the +122.6 ms wall gap.

Phase-level cross-check (llm.c ncu phases scaled to its 294.0 ms wall):

| phase | llm.mojo wall ms | llm.c wall-scaled ms | gap |
|---|---:|---:|---:|
| forward | 155.62 | 134.8 | +20.8 |
| backward | 244.05 | 145.2 | +98.9 |
| update | 16.99 | 14.0 | +3.0 |
| **total** | **416.66** | **294.0** | **+122.7** |

The gap is 81% backward / 17% forward — exactly what the TF32 asymmetry in
llm.c predicts: llm.c's *backward* dense GEMMs and all attention GEMMs go
through cuBLAS with TF32 tensor cores, while its *forward* dense GEMMs use
the slow hand-written SIMT kernel4 (so llm.mojo's forward, which at least
uses cuBLASLt SIMT kernels, only trails by ~21 ms there).

## Where the ~120 ms lives (hypotheses, ranked)

1. **TF32 disabled in llmm — the whole gap (~122 ms).** llm.mojo hardcodes
   `ComputeType.COMPUTE_32F` (`llmm/matmul.mojo:214`,
   `llmm/attention.mojo:4303`); every one of its 219 GEMM launches lands on
   SIMT (CUDA-core) kernels — `cutlass_80_simt_sgemm_*` + `magma_sgemmEx` —
   totaling 348.4 ms. llm.c enables `CUBLAS_TF32_TENSOR_OP_MATH` on sm≥80
   and its cuBLAS GEMMs run as `cutlass_80_tensorop_s1688gemm_*` (135.0 ms
   for the TF32 share). GEMM-family delta = 125.5 ms ≈ wall gap 122.6 ms;
   every non-GEMM family is already at parity. Fix: switch the two sites to
   `COMPUTE_32F_FAST_TF32` (numerics note: this is what llm.c fp32 and
   torch's float32_matmul_precision('high') already do — reference losses
   will shift).
2. **magma_sgemmEx_kernel (89.5 ms, 72 calls, 20.5% of step)** — under
   COMPUTE_32F, cuBLASLt's heuristic picks MAGMA-derived SIMT kernels for
   the most common shapes on GB10; these are the slowest per-call GEMMs in
   the capture (avg 1.24 ms). Expected to vanish with TF32 tensorop
   dispatch — worth re-profiling after the fix to confirm the heuristic
   doesn't stick to something odd on GB10.
3. **cuBLASLt::globalKernel split-K/epilogue launches (22.8 ms, 24 calls)**
   — auxiliary cuBLASLt kernels accompanying the SIMT GEMMs. Kernel
   selection (and these auxiliaries) will change wholesale under TF32;
   fold into the H1 re-measurement rather than chasing separately.
4. **Upside beyond parity:** with TF32 everywhere, llm.mojo should
   plausibly *beat* llm.c fp32, because llm.c leaves its forward dense
   GEMMs on the hand-written SIMT kernel4 (79.0 ms/step, 25.7% of its
   step) — llm.mojo would get tensorop kernels for those same shapes.
   Rough bound: mojo GEMM 348 ms → TF32 at the llm.c-observed tensorop
   rates ≈ 150–190 ms → step ≈ 250–290 ms vs llm.c's 294 ms.
5. **Nothing else is worth touching for this goal.** Attention
   softmax fwd+bwd (27.4 vs 27.1 ms), optimizer (16.8 vs 14.7), classifier
   (8.2 vs 9.4) are all at parity; non-GEMM totals differ by 2.5 ms.

## AI use statement

Written with AI assistance (Claude Code / Sonnet agent), directed by Evan Owen.
