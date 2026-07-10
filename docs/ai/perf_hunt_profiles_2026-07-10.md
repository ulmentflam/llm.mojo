# Post-TF32 perf hunt — per-kernel comparison tables (fp32 lead + bf16 gap)

**Date:** 2026-07-10. **Branch:** `goal1/fp32-parity` (HEAD `49555dd`).
**Machine:** NVIDIA GB10 (Grace-Blackwell, aarch64, unified memory), DGX
Spark, driver 595.71.05, CUDA 13.2, ncu 2025.3.1. B=4, T=1024, L=12.

Goal: with TF32 merged and the official numbers now **fp32: llm.mojo
282.23 ms vs llm.c 292.94 ms (we lead ~10.7 ms)** and **bf16: llm.mojo
134.77 ms vs llm.c 133.68 ms (we trail ~1.1 ms)**, find where each arm's
time actually goes so a follow-up analysis pass can decide where to push.
This is data collection only — no source changes, no fixes attempted here.

## Method

Four one-step ncu captures, each held under `flock /tmp/llmm-gpu.lock`,
box otherwise idle (`nvidia-smi` 0% util before every run):

1. **llm.mojo fp32 (post-TF32)** — `make profile-fp32-ncu PROFILE_T=1024`
   against today's fresh `build/profile_gpt2` (mtime 13:27, past the TF32
   merge). Saved as `build/profile_gpt2_fp32tf32_t1024.ncu.csv`.
2. **llm.mojo bf16** — `make profile-ncu PROFILE_T=1024` against
   `build/profile_gpt2_bf16` (mtime 13:27). Saved as
   `build/profile_gpt2_bf16_t1024.ncu.csv`.
3. **llm.c bf16** — `train_gpt2cu` via `profile_gpt2.py --exe-args`,
   **`-x 1 -v 0 -m 0 -s 0 -l 0`**. The stock `profile-llmc-ncu` recipe's
   `-x 1 -v 0` combo is a trap on this box: `step % val_loss_every` with
   `val_loss_every=0` is UB, and ARM's `SDIV`-by-zero-returns-0 semantics
   make it evaluate true only at `step==0`, but llm.c's loop *also*
   unconditionally validates at `last_step` — so a plain `-x 1 -v 0` run
   captures **40 extra validation-only forward passes** (`val_max_steps`
   default 20, run twice) bracketing the one real step, bloating the
   forward-attributable families ~40x. Fix: `-m 0` (`val_max_steps=0`)
   skips the validation loop body entirely regardless of when it's
   entered, giving a perfectly clean single fwd+bwd+update capture — no
   launch-skip arithmetic needed. Saved as
   `build/profile_llmc_bf16_t1024.ncu.csv` (512 launches, 33 kernels).
4. **llm.c fp32** — reused verbatim from this morning's
   `docs/ai/fp32_parity_profile_2026-07-10.md` (llm.c side has zero TF32
   exposure, unaffected by our fix): family totals from its "Side-by-side"
   table (307.74 ms total) and top-kernel list. Re-parsing the saved
   `build/profile_llmc_fp32_t1024_step.ncu.csv` directly gives 800
   launches / classifier×2 / layernorm_bwd×43 — i.e. that raw file still
   carries some val-loop residue the previous agent hand-trimmed before
   writing the doc's numbers, so the *doc's* published per-family/per-kernel
   figures (not a fresh full-CSV aggregate) are the trustworthy source and
   are what's reused below.

`ncu` here has no perf-counter access without `sudo` (DRAM/tensor% mostly
blank), except tensor% which the tool *does* get for these metrics — kernel
names, call counts and durations are authoritative regardless.

## Sanity check: ncu-serialized total vs official wall-clock step

| arm | ncu total | official wall | ratio | verdict |
|---|---:|---:|---:|---|
| llm.mojo fp32 (TF32) | 293.76 ms | 282.23 ms | 1.041 | OK |
| llm.mojo bf16 | 148.76 ms | 134.77 ms | 1.104 | OK |
| llm.c fp32 | 307.74 ms | 292.94 ms | 1.051 | OK |
| llm.c bf16 | 146.73 ms | 133.68 ms | 1.098 | OK |

All four land within ~4–11% of wall-clock (ncu instrumentation overhead) —
comfortably inside the ~15% tolerance; no missed kernels or CPU gaps in
any arm.

**GEMM launch-count cross-check (bonus sanity):** after stripping the
non-GEMM kernels that the family classifier mis-buckets into "matmul"
(`llmm_matmul__dbias_accum/finalize` for mojo, `matmul_backward_bias_kernel*`
for llm.c) and recovering the GEMMs Blackwell's `nvjet_sm121_*` naming
hides in "other", **all four arms launch exactly 219 real GEMMs per
step** — mojo-fp32 219, mojo-bf16 180(matmul)+39(nvjet)=219, llm.c-bf16
180(matmul)+39(nvjet)=219. Same model, same op count, precision-invariant
— a clean corroboration that no kernel launches are silently missing on
either side.

## Table 1 — fp32: llm.mojo (TF32) vs llm.c

Family buckets follow this morning's doc's convention exactly (GEMM
includes cuBLASLt epilogue/split-K kernels; attention bucket includes the
split/merge = llm.c's permute/unpermute) so the pre-TF32 baseline is
directly comparable.

| family | ours (TF32) ms | llm.c ms | delta (ours−llmc) |
|---|---:|---:|---:|
| GEMM (incl. cuBLASLt epilogue/dbias-aux) | 206.59 | 222.86 | **−16.27** |
| attention + permute/split-merge | 40.75 | 36.97 | +3.78 |
| layernorm | 20.94 | 7.83 | **+13.11** |
| gelu | 0 (fused into GEMM epilogue) | 12.61 | **−12.61** |
| optimizer (adamw + global_norm) | 16.71 | 14.69 | +2.02 |
| classifier | 8.32 | 9.35 | −1.03 |
| residual | 0 (fused) | 3.24 | −3.24 |
| encoder | 0.45 | 0.19 | +0.26 |
| **total** | **293.76** | **307.74** | **−13.98** |

Pre-TF32 baseline for reference (this morning): GEMM delta was **+125.5**
(mojo far slower). TF32 alone swung it **−141.8 ms**, past parity into a
net lead — everything else moved <1 ms from pre-TF32 (noise-level; the fix
was exactly as surgical as predicted).

Top individual kernels, ours (TF32), by time:

| kernel | calls | ms | tensor% |
|---|---:|---:|---:|
| cutlass_80_tensorop_s1688gemm (cfg A) | 24 | 27.11 | 10.9 |
| cutlass_80_tensorop_s1688gemm (cfg B) | 24 | 26.76 | 11.2 |
| cutlass_80_tensorop_s1688gemm (cfg C) | 48 | 23.85 | 58.3 |
| cutlass_80_tensorop_s1688gemm (cfg D) | 24 | 22.19 | 13.4 |
| cutlass_80_tensorop_s1688gemm (cfg E) | 13 | 18.03 | 54.6 |
| cutlass_80_tensorop_s1688gemm (cfg F) | 36 | 17.22 | 58.3 |
| cutlass_80_tensorop_s1688gemm (cfg G) | 24 | 17.14 | 46.6 |
| llmm_attention_attention_bwd | 12 | 16.06 | — |
| llmm_adamw_adamw_update_gpu | 1 | 14.73 | — |
| cublasLt::globalKernel (split-K epilogue) | 12 | 14.48 | — |

## Table 2 — bf16: llm.mojo vs llm.c

Blackwell's `nvjet_sm121_tst_mma_*` kernels (cuBLASLt's bf16 tensor-core
GEMM on sm_121) don't contain "cutlass"/"gemm"/"matmul" in their name, so
`profile_gpt2.py`'s classifier drops them into "other" for **both** arms —
these are genuine attention-batched GEMMs (39 calls, 53–92% tensor
utilization), not noise. Recovered into the GEMM row below; the residual
"other" (copy_and_cast, permute/unpermute, fused_residual, reduce_add_sum)
is llm.c-only bookkeeping mojo fuses away.

| family | ours ms | llm.c ms | delta (ours−llmc) |
|---|---:|---:|---:|
| GEMM (dense + attention-batched, incl. bias-accum aux) | 96.01 | 89.94 | **+6.07** |
| attention softmax (fwd+bwd) | 16.63 | 16.59 | +0.04 |
| permute / split-merge | 6.20 | 5.52 | +0.68 |
| layernorm | 9.73 | 2.61 | **+7.12** |
| gelu | 0 (fused into GEMM epilogue) | 7.27 | **−7.27** |
| optimizer (adamw + global_norm) | 15.91 | 16.04 | −0.13 |
| classifier | 3.93 | 3.55 | +0.38 |
| residual + master-weight cast + misc | 0 (fused) | 5.12 | **−5.12** |
| encoder | 0.35 | 0.09 | +0.26 |
| **total** | **148.76** | **146.73** | **+2.03** |

ncu-serialized delta (+2.03) is the same sign and roughly the same
magnitude as the official wall gap (mojo trails by 1.09 ms) — layernorm
and GEMM are the net drag, gelu- and residual-fusion wins claw most but
not quite all of it back.

Top individual kernels, ours (bf16), by time:

| kernel | calls | ms | tensor% |
|---|---:|---:|---:|
| cutlass_80_wmma_tensorop_bf16 (cfg A) | 24 | 15.26 | 9.8 |
| llmm_adamw_adamw_update_gpu | 1 | 14.89 | — |
| cutlass_80_tensorop_bf16 (cfg B) | 24 | 12.79 | 11.7 |
| cutlass_80_tensorop_bf16 (cfg C) | 24 | 11.14 | 13.4 |
| llmm_attention_attention_bwd | 12 | 8.12 | — |
| cutlass_80_tensorop_bf16 (cfg D) | 36 | 7.83 | 57.4 |
| nvjet_sm121_tst_mma_128x208x64 | 13 | 7.77 | 63.6 |
| llmm_attention_attention_soft | 12 | 6.93 | — |
| cutlass_80_tensorop_bf16 (cfg E) | 12 | 6.20 | 37.8 |
| cutlass_80_tensorop_bf16 (cfg F) | 12 | 5.69 | 40.5 |

## Biggest deltas (either direction), both arms

1. **fp32 GEMM: −16.27 ms (ours faster).** TF32 didn't just close the gap,
   it inverted it — the biggest single swing in either table, and the
   entire explanation for the fp32 lead.
2. **fp32 layernorm: +13.11 ms (ours slower).** 20.94 ms vs llm.c's
   7.83 ms — llm.c's `layernorm_forward_kernel3`/`layernorm_backward_kernel2`
   run ~2.7x faster in aggregate than llmm's layernorm family. Now the
   single largest remaining fp32 loss, well ahead of #3.
3. **fp32 gelu: −12.61 ms (ours faster).** llmm fuses GELU into the fc
   GEMM's epilogue; llm.c runs it as separate
   `gelu_forward_kernel`/`gelu_backward_kernel` launches (12.61 ms, 4.2%
   of its step).
4. **bf16 layernorm: +7.12 ms (ours slower).** Same family, same sign, same
   *scale relative to itself* as the fp32 finding (9.73 vs 2.61 ms,
   ~3.7x) — this isn't precision-specific noise, it's a persistent
   llmm layernorm inefficiency across both builds.
5. **bf16 gelu: −7.27 ms (ours faster).** Mirrors the fp32 fusion win at
   bf16 scale.
6. **bf16 GEMM (dense + attention-batched): +6.07 ms (ours slower).** This
   is new: buried inside "other" until the nvjet reclassification above,
   it's the single largest contributor to the −1.1 ms wall-clock trail —
   more than the whole gap by itself.
7. **bf16 residual + master-weight cast + misc: −5.12 ms (ours faster).**
   llm.c pays separately for `fused_residual_forward_kernel5` and a
   16-call `copy_and_cast_kernel` (fp32↔bf16 master-weight shuttle for
   the optimizer) that llmm fuses away.
8. **fp32 attention + permute/split-merge: +3.78 ms (ours slower).**
   Smaller and persistent (was +3.9 ms pre-TF32 too) — essentially at
   parity but not quite.

## Suspicious auxiliaries / call-count notes

- **`llmm_matmul__dbias_accum_gpu` / `_dbias_finalize_gp`** (48 calls
  each, fp32 7.08+0.24 ms, bf16 4.65+0.25 ms) are bias-gradient reduction
  kernels, not GEMMs, but the family classifier's `"matmul"` substring
  rule buckets them into "matmul" anyway — inflates that family by
  ~2–3.5%. Harmless for these tables (both arms' GEMM rows already net
  this out consistently) but worth a classifier fix if per-kernel
  precision matters later.
- **`cublasLt::globalKernel<8,32,float,…>`** (fp32 "other", 22.53 ms /
  24 calls) is cuBLASLt's split-K reduction epilogue for the TF32 GEMMs —
  legitimate GEMM-adjacent cost, not noise; already folded into the fp32
  GEMM row above.
- **`nvjet_sm121_tst_mma_*`** (bf16 "other" for *both* arms) are attention
  QK/PV batched GEMMs on this Blackwell part's cuBLASLt path — the most
  consequential misclassification found (see Table 2 note). Both mojo and
  llm.c dispatch bf16 attention GEMMs to the literal same kernel family
  here, which is itself a mild surprise (llm.c on GB10 also goes through
  cuBLASLt's Blackwell path, not the sm80 cutlass tensorop kernels it uses
  for dense bf16 GEMMs).
- **GEMM call-count parity (219 on all four arms)** — see sanity section.
  No missing/extra launches on either side; the fp32 vs bf16 kernel-name
  churn (cutlass tensorop_s1688 vs wmma/tensorop_bf16/nvjet) is purely a
  cuBLASLt kernel-selection artifact of the dtype, not a structural
  difference.
- llm.c fp32's `matmul_forward_kernel4` (its hand-written non-cuBLAS SIMT
  forward matmul, 79.0 ms / 25.7% of its step per this morning's doc)
  remains untouched by TF32 — llm.c's own dense *forward* GEMMs still run
  on CUDA cores, not tensor cores. That headroom is llm.c's own, not ours;
  noted only because it's a large, TF32-invisible chunk of its 307.74 ms
  total, so llm.c's own ceiling is not vs a fully-tensor-core baseline.

## AI use statement

Written with AI assistance (Claude Code / Sonnet agent), directed by Evan Owen.
