# Post-TF32 perf hunt — ranked optimization plan (flip bf16, extend fp32)

**Date:** 2026-07-10. **Branch:** `goal1/fp32-parity` (HEAD `d6bb973`).
**Machine:** NVIDIA GB10 (Grace-Blackwell, aarch64, unified LPDDR5X
~273 GB/s), CUDA 13.2, ncu 2025.3.1. B=4, T=1024, L=12.

Companion to `docs/ai/perf_hunt_profiles_2026-07-10.md` (the 4-arm ncu data
collection). This doc **root-causes** each delta by reading *both*
implementations and turns them into a ranked, implementable plan. Read-only
analysis — no code was changed and no GPU was run.

Official wall-clock standings: **fp32 llm.mojo 282.23 ms vs llm.c 292.94 ms
(we lead 10.7 ms)**; **bf16 llm.mojo 134.77 ms vs llm.c 133.68 ms (we trail
1.1 ms)**. ncu-serialized totals: fp32 293.76 vs 307.74; bf16 148.76 vs
146.73.

## TL;DR

- **The layernorm gap is 100% in the BACKWARD, both precisions.** The
  forward is already at parity (bf16 +0.09 ms, fp32 +0.03 ms). The profile
  doc's headline "layernorm +13.11/+7.12" is a *family-bucketing artifact*:
  it compares our residual-add-fused LN forward against llm.c's LN that
  excludes the residual add. Matched op-for-op, the real layernorm loss is
  **fp32 +9.88 ms / bf16 +5.40 ms, entirely in the backward.**
- **Root cause:** our LN backward runs **four kernels** (residual-grad
  broadcast, d_input, dgamma/dbeta accum via atomics, dgamma/dbeta finalize)
  ≈ 13 element-passes over `[B,T,C]`. llm.c runs **one** fused
  `layernorm_backward_kernel10` ≈ 5 passes. On bandwidth-bound GB10, passes
  are the currency.
- **The bf16 "GEMM +6.07" is also a bucketing artifact.** Split: dense
  cutlass +4.11, attention-batched nvjet **−0.13 (parity)**, dbias +2.09.
  The dense +4.11 is *linked* to our gelu-fusion win (−7.27) — our gelu
  lives in the GEMM epilogue. GEMM+gelu combined we are **−3.15 ms faster**.
  There is no independent dense-GEMM lever; the attention GEMMs already
  match llm.c kernel-for-kernel (confirms the `USE_LT_ATTN` NEUTRAL note).
- **Wave-1 = two parallel worktrees applying the same fused-row-reduction
  technique:** (1) LN-backward single-kernel fusion, (2) matmul-dbias
  fusion. Together they project **bf16 → a ~4.7 ms wall lead (flip)** and
  **fp32 → a ~20 ms wall lead (up from 10.7)**.

## Method note — correcting the family buckets

The `profile_gpt2.py` family classifier mis-attributes across
implementations in two ways that inflate the headline layernorm/GEMM
deltas. This analysis re-buckets by **operation**, verified against the raw
`build/*.ncu.csv` `gpu__time_duration.sum` rows (aggregated per kernel
name, one-step captures confirmed by adamw ×1 / encoder ×1):

1. **Residual add.** llm.c bf16 fuses residual+LN in
   `fused_residual_forward_kernel5`, which the classifier buckets as
   *residual*; ours fuses the same add into our LN forward kernel, bucketed
   as *layernorm*. So the doc credits llm.c a −1.63 ms "residual" while
   charging us the same work under "layernorm". Matched, the forwards tie.
2. **GELU.** Our GELU is fused into the linear-GEMM cuBLASLt epilogue (0
   separate ms); llm.c runs it as separate `gelu_forward/backward` (7.27 ms
   bf16 / 12.61 fp32). The classifier therefore shows our *dense GEMM* as
   "+4.11 slower" while crediting a "−7.27 gelu" — but the gelu cost is
   *inside* our GEMM number. GEMM+gelu combined, we lead.

Kernel launch-ID ordering (forward = low IDs, backward = high IDs) was used
to split our two same-named `layernorm_fused` kernel hashes: the grid-1536
hash (IDs 9–133) is the **forward**; the grid-3072 hash (IDs 149–524) is the
backward **`residual_grad_broadcast`** — a distinct fourth backward kernel,
not the forward.

## Operation-matched deltas — bf16 (ncu ms, ours − llm.c)

| operation | ours | llm.c | Δ | note |
|---|---:|---:|---:|---|
| dense GEMM (cutlass, incl. fused gelu+bias epilogue) | 68.82 | 64.70 | +4.11 | linked to gelu below |
| gelu | 0 | 7.27 | −7.27 | fused into our GEMM epilogue |
| → **GEMM+gelu combined** | 68.82 | 71.97 | **−3.15** | **ours faster** |
| attention-batched GEMM (nvjet) | 22.29 | 22.42 | **−0.13** | parity, same kernels |
| matmul dbias | 4.90 | 2.81 | **+2.09** | ours 96 launches vs 48 |
| attention softmax + P/dS | 16.63 | 16.59 | +0.04 | parity |
| LN forward (+residual add) | 1.75 | 1.66 | +0.09 | parity |
| **LN backward** | **7.98** | **2.58** | **+5.40** | 4 kernels vs 1 |
| permute / split-merge | 6.20 | 5.52 | +0.68 | at floor |
| adamw | 14.89 | 14.77 | +0.12 | parity |
| global-norm | 1.01 | 1.27 | −0.26 | ours faster |
| classifier | 3.93 | 3.55 | +0.38 | ~parity |
| master-weight copy/cast | 0 | 3.29 | −3.29 | we avoid the fp32↔bf16 shuttle |
| encoder | 0.35 | 0.09 | +0.26 | tiny |
| misc reduce/memset | 0 | 0.19 | −0.19 | |
| **total** | **148.76** | **146.73** | **+2.03** | |

The +2.03 ncu gap is dominated by **LN-backward +5.40** and **dbias +2.09**;
everything else is either a win, parity, or at the bandwidth floor.

## Operation-matched deltas — fp32/TF32 (ncu ms)

llm.c fp32 per-op from the trimmed figures in
`fp32_parity_profile_2026-07-10.md` (its raw CSV carries val-loop residue);
ours from the clean `profile_gpt2_fp32tf32_t1024.ncu.csv`.

| operation | ours | llm.c | Δ | note |
|---|---:|---:|---:|---|
| GEMM (dense+epilogue+dbias, TF32) | 206.59 | 222.86 | **−16.27** | TF32 win, incl. dbias 7.32 |
| gelu | 0 | 12.61 | −12.61 | fused |
| attention softmax + P/dS | 30.08 | — | — | fp32-heavy scratch |
| permute / split-merge | 10.68 | — | — | |
| → attention + permute total | 40.76 | 36.97 | +3.78 | fp32-inherent bandwidth |
| LN forward (+residual add) | 5.51 | 5.48 | +0.03 | parity |
| **LN backward** | **15.43** | **5.55** | **+9.88** | 4 kernels vs 1 |
| adamw | 14.73 | 14.69 | +0.04 | parity |
| global-norm | 1.98 | (in opt) | ~+2.0 | fp32 = 2× grad bytes |
| classifier | 8.32 | 9.35 | −1.03 | ours faster |
| encoder | 0.45 | 0.19 | +0.26 | tiny |
| **total** | **293.76** | **307.74** | **−13.98** | |

The fp32 lead is TF32 (GEMM −16.27) + gelu fusion (−12.61). The single
largest thing dragging *against* the lead is **LN-backward +9.88**.

### LN backward decomposition (our 4 kernels)

| kernel (our name) | role | fp32 ms ×24 | bf16 ms ×24 |
|---|---|---:|---:|
| `residual_grad_broadcast` (grid 3072) | seed d_inp1/d_inp2 += incoming residual grad | 4.43 | 2.16 |
| `layernorm_bwd_residual_gpu` (bwd_r) | d_input, then d_inp1/d_inp2 += d_input (RMW) | 7.25 | 3.18 |
| `_ln_dparam_accum_gpu` (atomics) | dgamma/dbeta, re-reads dout+input | 3.54 | 2.48 |
| `_ln_dparam_finalize_gpu` | scratch → grads, re-zero | 0.09 | 0.09 |
| plain `layernorm_bwd` (final LN, ×1) | — | 0.12 | 0.06 |
| **total LN backward** | | **15.43** | **7.98** |
| llm.c `layernorm_backward_kernel10` (1 kernel) | d_input + dweight + dbias + resid, fused | **5.55** | **2.58** |

Both `residual_grad_broadcast` and `bwd_r` do a `+=` RMW into d_inp1/d_inp2
(two passes doing the same broadcast). `_ln_dparam_accum` re-reads the full
`dout`+`input` planes that `bwd_r` already streamed. That redundancy is the
whole gap.

---

## Ranked candidate list

### #1 — LN backward single-kernel fusion  ★ wave-1

- **Root cause.** We split the LN backward into 4 kernels (~13 passes over
  `[B,T,C]`); llm.c does it in one `layernorm_backward_kernel10` (~5 passes)
  with a shared-memory block reduction for dgamma/dbeta and a flag-gated
  finalize inside the same launch. On bandwidth-bound GB10 the extra passes
  *are* the cost. Forward is already at parity, so this is the whole
  layernorm story.
- **Expected saved:** **fp32 −7.4 ms (best −9.0), bf16 −4.0 ms (best −5.0).**
  (Model: fold `residual_grad_broadcast` into `bwd_r` by computing
  `d_inp1 = resid_in + d_input` directly — eliminates a full RMW pass and
  the separate broadcast kernel; merge `_ln_dparam_accum` into `bwd_r` via a
  kernel10-style shared reduction — eliminates the re-read of dout+input.
  Target ~5 passes ≈ 4.0/8.0 ms; llm.c's floor is 2.58/5.55.)
- **Implementation.** `llmm/layernorm.mojo`. Rewrite
  `layernorm_fused_residual_bwd` (GPU branch, lines 2214-2297) to launch a
  single fused kernel modeled on
  `third_party/llm.c/llmc/layernorm.cuh:233 layernorm_backward_kernel10`:
  - Extend `_layernorm_bwd_residual_gpu` (line 1435) to (a) take the
    incoming-residual-grad pointer and write `d_inp1/d_inp2 = resid_in +
    d_input` in pass 2 (drop the separate `residual_grad_broadcast` /
    `_layernorm_fused_residual_bwd_broadcast_tile` call), and (b) accumulate
    dgamma/dbeta into shared memory in the same pass 1/2, replacing
    `_ln_dparam_accum_gpu` (line 1691) + `_ln_dparam_finalize_gpu`
    (line 1727). Use llm.c's block-partial-to-scratch + `atomicInc`-flag
    "last block finalizes" pattern, or keep the existing persistent
    `_ln_dparam_scratch` (line 1670) and a tiny finalize — either removes
    the separate full-tensor re-read, which is the win.
  - Replace the current non-deterministic `Atomic.fetch_add` dgamma/dbeta
    accumulation with the deterministic shared/tree reduction — this
    *improves* bit-stability (see gate).
  - dtype-generic: the same kernel serves fp32 and bf16 (both precisions
    win). `layernorm_bwd` (non-fused, used only for layer-0 LN1 / final LN,
    ×1 each) can share the same fused kernel or be left as-is (negligible).
- **Gate.** `make verify-gpu` (strict-IEEE, TF32 off) checking **dinput,
  dgamma, dbeta separately** against PyTorch — NOT a flat atol (see
  `weak-gates-overrule-nothing`: a flat atol=2.0 previously buried three
  backward bugs; require the per-tensor rel/abs tolerances). Plus the bf16
  loss-trajectory twin-run / cross-build bit-stability protocol — the
  reduction order changes, so re-baseline the reference trajectory; a
  deterministic reduction should make it *more* stable than today's atomics.
- **Risk.** Medium. It is a genuine kernel rewrite with shared-memory
  reduction and a flag-gated finalize (delicate). But the win is pure
  pass-count reduction (bandwidth), *not* the block/warp remapping that the
  history flagged as a dead end — the "warp-per-row port was slower" and
  "LN SMEM-cache neutral" negatives were all about the **forward**; the
  backward is explicitly noted as "the untried target". No arithmetic
  change beyond reduction order.
- **Parallelizable with:** #2 (different file), #3, #4. No FP8 collision
  (`goal2/fp8-training` touches `matmul.mojo` only; LN is untouched).

### #2 — matmul dbias fusion (row-reduction)  ★ wave-1

- **Root cause.** Our `matmul_bias_bwd` GPU path runs **two** kernels
  (`_dbias_accum_gpu` row-block partials + `_dbias_finalize_gpu`) ×48 = 96
  launches (bf16 4.90 ms / fp32 7.32 ms). llm.c's
  `matmul_backward_bias_kernel9` (`llmc/matmul.cuh:17`) does it in **one**
  launch ×48 = 2.81 ms bf16 via a cooperative warp/block reduction that
  writes dbias directly (or through a single `reduce_add_sum`). Same "reduce
  dout over the B·T rows" pattern as #1. Our code already uses the good
  contention-free partial-buffer layout (comment at
  `matmul.mojo:932`, "Matches llm.c's") — the loss is the **extra finalize
  launch + a per-call re-read**, not the algorithm.
- **Expected saved:** **bf16 −2.0 ms** (4.90 → ~2.9, match llm.c), **fp32
  −3.0 ms** (7.32 → ~4.3). fp32 dbias currently sits *inside* the −16.27
  GEMM win, so this deepens an existing fp32 lead and directly cuts the bf16
  gap.
- **Implementation.** `llmm/matmul.mojo`: fuse `_dbias_accum_gpu`
  (line 909) + `_dbias_finalize_gpu` (line 937) into a single
  `matmul_backward_bias_kernel9`-style kernel (grid.y row-blocks +
  cooperative reduction + flag-gated last-block finalize), eliminating the
  second launch and the scratch round-trip. Reuse the exact reduction helper
  written for #1 if factored out.
- **Gate.** `make verify-gpu` per-tensor tolerance on **dbias** vs PyTorch;
  bf16 twin-run trajectory. Same reduction-order re-baseline as #1.
- **Risk.** Low-medium. Well-understood pattern, our layout is already
  correct. **FP8 interaction: `matmul.mojo` is also edited by
  `goal2/fp8-training`.** dbias is a bias-*gradient* reduction, not a GEMM
  operand, so it should not collide with fp8 (GEMM-operand-only), but it is
  the same file — coordinate the merge / rebase with the FP8 owner. Flagged.
- **Parallelizable with:** #1, #3, #4 (share the reduction helper with #1).

### #3 — fp32 attention + permute (+3.78) / global-norm (+2.0)  — wave-2, low ROI

- **Root cause.** Both are **fp32-inherent bandwidth**, not structural. Our
  attention softmax + P/dS passes are fp32 (11.28 softmax + 16.06 P/dS +
  2.74 D = 30.08 ms) — ~2× their bf16 selves (16.63) because the fp32 build
  keeps fp32 `[B,nh,T,T]` scratch; bf16 narrows it (softmax/pds are at
  **parity** in bf16, +0.04). llm.c fp32 also uses an fp32 attention matrix
  but is 3.78 lighter across softmax+permute. Global-norm reads all 124M
  grad elements; fp32 = 2× bytes → 1.98 vs bf16 1.01.
- **Expected saved:** small and hard. The history documents softmax
  512-thread (worse), split/merge x128 vectorize (worse) as measured dead
  ends — these are at the L2/DRAM floor. Only attackable by a fused
  tensor-core flash kernel (proven a **dead end on this hardware** — see the
  extensive PoC log) or fusing softmax into the QKᵀ epilogue (bf16 already
  did the cheap version). **Do not spend wave-1 here.**
- **Note.** fp32 already leads by 10.7 ms; this delta does not threaten the
  lead, and bf16 (the arm that needs the win) is already at parity here.
- **Risk.** High effort / low return. Deprioritize.

### #4 — auxiliary launch-count trims  — fold into #1/#2

- The `_ln_dparam_finalize` (0.09 ms) and `_dbias_finalize` (0.25 ms)
  kernels vanish for free once #1 and #2 fuse the finalize into the main
  kernel (flag-gated last block). No separate work item — this is a
  *consequence* of #1/#2, listed so it is not double-counted as a saving.
- The `profile_gpt2.py` classifier's `"matmul"`-substring rule mis-buckets
  `_dbias_accum/finalize` into GEMM and drops `nvjet_*` into "other". Worth
  a one-line classifier fix so future profiles read correctly, but it is a
  tooling nicety with **zero** runtime effect.

---

## Recommended wave-1 dispatch

Two independent worktrees, both applying the **same fused row-reduction
technique** (compute-and-reduce-over-rows in a single kernel with a
flag-gated finalize):

| worktree | candidate | file | expected fp32 / bf16 | fp8 collision |
|---|---|---|---:|---|
| `goal1/ln-bwd-fuse` | #1 LN-backward fusion | `llmm/layernorm.mojo` | −7.4 / −4.0 ms | none |
| `goal1/dbias-fuse` | #2 matmul-dbias fusion | `llmm/matmul.mojo` | −3.0 / −2.0 ms | coordinate |

They touch disjoint files and can land in either order. Recommend building
the shared reduction helper (block-partial → scratch → `atomicInc`-flag
finalize) once in #1 and reusing it in #2. Both must clear the **per-tensor
strict-IEEE verify-gpu gate** (not a flat atol) and the **bf16 twin-run /
cross-build loss-trajectory** protocol with a re-baselined reference.

Skip #3/#4 as wave-1: #3 is fp32-only and at the bandwidth floor (won't
threaten the existing fp32 lead, doesn't help the bf16 arm that needs it);
#4 falls out of #1/#2 for free.

## Projected post-fix step times

Using conservative saves (best-case in parentheses), scaling ncu→wall by the
measured ~0.9 ratio:

| arm | metric | today | after #1 | after #1+#2 |
|---|---|---:|---:|---:|
| **bf16** | ncu ms | 148.76 | 144.8 (143.8) | **142.8 (141.8)** |
| | vs llm.c 146.73 | +2.03 | −1.9 (−2.9) | **−3.9 (−4.9)** |
| | wall ms (est.) | 134.77 | ~131.2 | **~129.2** |
| | vs llm.c 133.68 | +1.09 | ~−2.5 | **~−4.5 (flip)** |
| **fp32** | ncu ms | 293.76 | 286.4 (284.8) | **283.4 (281.8)** |
| | vs llm.c 307.74 | −13.98 | −21.3 | **−24.3** |
| | wall ms (est.) | 282.23 | ~275.5 | **~272.5** |
| | vs llm.c 292.94 | −10.71 | ~−17.4 | **~−20.4** |

**Bottom line:** candidate #1 alone flips bf16 to a clear ~2.5 ms wall win
and pushes the fp32 lead past 17 ms; adding #2 takes bf16 to ~4.5 ms and
fp32 past 20 ms. Both are the same low-risk bandwidth lever (fuse the
row-reduction passes) that llm.c already exploits with its single-kernel
backward.

## AI use statement

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
