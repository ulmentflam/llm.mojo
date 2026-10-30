# Muon + MuonClip implementation plan (2026-10-29)

Status: **plan only, nothing implemented.** Follow-up to the Tier-B verdict in
[`speedrun_techniques_research.md`](speedrun_techniques_research.md) (B1: Muon
is the single biggest speedrun win, but it breaks the llm.c AdamW-parity
benchmark). The resolution here: Muon ships as an **opt-in** optimizer
(`-opt muon`, default stays `adamw`), so the parity track is untouched and a
"fastest-to-loss" track becomes possible.

## 1. What we are building

**Muon** (Jordan et al., 2024): for every 2-D weight matrix, SGD with Nesterov
momentum whose momentum matrix is replaced by its nearest semi-orthogonal
matrix, computed with 5 Newton–Schulz (NS) iterations in bf16.

```
M  = mu * M + G                       # fp32 momentum (reuses m_memory)
U  = G + mu * M                       # Nesterov
O  = NS5(U / (||U||_F + 1e-7))        # bf16, 5 steps, (a,b,c) = (3.4445, -4.7750, 2.0315)
W  = W - lr * (0.2 * sqrt(max(m, n)) * O + wd * W)   # Moonlight scaling + decoupled wd
```

The `0.2 * sqrt(max(m, n))` factor is the Moonlight (Kimi) RMS-matching rule;
it lets Muon share AdamW's `-l` learning rate and `-c` weight decay instead of
needing a second tuned LR. modded-nanogpt instead uses a separate LR
(~0.02–0.05) and `max(1, m/n)^0.5`; we start from Moonlight because it needs
no new hyperparameter sweep, and keep the alternative behind a constant.

**Which tensors.** Muon applies only to the hidden-layer matrices; everything
else stays on AdamW exactly as today (same kernel, same `m`/`v` slots):

| tensor (per layer, row-major `[out, in]`) | shape at 124M | optimizer | note |
|---|---|---|---|
| `qkv_weight` split into Q, K, V row blocks | 3 × `[C, C]` | Muon | orthogonalize each block separately (modded-nanogpt has separate Q/K/V Linears) |
| `attn_proj_weight` | `[C, C]` | Muon | |
| `fc_weight` | `[4C, C]` | Muon | m > n: NS runs on the transpose so the Gram is `C×C` |
| `proj_weight` | `[C, 4C]` | Muon | |
| `wte`, `wpe`, all biases, all LN gamma/beta | – | AdamW | unchanged |

**MuonClip** (Kimi K2, 2025) = Muon + **QK-clip**: after each optimizer step,
if any head's max pre-softmax attention logit `S_max,h` exceeded `tau`
(K2 uses 100), rescale that head's Q and K projections by
`gamma_h^alpha` and `gamma_h^(1-alpha)` with `gamma_h = tau / S_max,h`,
`alpha = 0.5`. For GPT-2 that means scaling rows `h*hd:(h+1)*hd` of the Q and K
blocks of `qkv_weight` **and the matching `qkv_bias` entries** (the logit is
`(xWq+bq)·(xWk+bk)`, so the bias has to scale too or the clip is wrong). It is
a no-op until a head crosses `tau`, which a 124M GPT-2 at lr 6e-4 likely never
does; it is insurance for the larger / longer runs, and cheap to carry.

## 2. How it maps onto the existing code

Everything below is grounded in the current tree (commit `6574ea2`).

- **Flat parameter buffer + fixed tensor order** (`train_gpt2.mojo:450`
  `Parameters`, `_compute_param_sizes` at `:1978`): each tensor is `L`
  contiguous per-layer matrices. All `L` copies of one tensor type are
  equally strided, so NS can run as **one strided-batched cuBLASLt call per
  tensor group** (`_matmul_cublaslt` already takes `batch_count` / `stride_*`,
  `llmm/matmul.mojo:464`). Six groups: Q, K, V, attn_proj, fc, proj; each with
  `batch_count = L`.
- **Momentum buffer**: `m_memory` (fp32, `train_gpt2.mojo:1276`) is reused as
  Muon's `M` for the Muon-owned ranges. `v_memory` stays allocated but unused
  there (0.3 GB wasted at 124M; freeing it is a later memory follow-up).
  Checkpoint state files (`llmm/checkpointing.mojo`, 256-int header) keep the
  same `m`/`v` blob layout; add one header slot `optimizer_kind` (0 adamw,
  1 muon) and refuse to resume across a mismatch.
- **Master weights / bf16 store**: the final `W -= lr(...)` is an elementwise
  kernel structurally identical to `_adamw_update` (`llmm/adamw.mojo:97`), so
  the fp32-master, `has_master`, and `LLMM_SR_MASTER` stochastic-rounding
  store logic is lifted into a shared helper rather than duplicated.
- **Per-matrix Frobenius norm**: `global_norm_squared_gpu`
  (`llmm/global_norm.mojo:156`) already supports `num_slices × stride`
  reductions via `block_idx.y`; call it once per group with `stride = m*n`
  to get all `L` norms in one launch.
- **Grad clipping** (`train_gpt2.mojo:6748`): keep applying the global
  `grad_scale` before the momentum update. Orthogonalization makes Muon almost
  scale-invariant, so `-gc` is harmless; it stays for AdamW-side tensors.
- **Attention logit max for QK-clip**: `_attention_softmax_causal_gpu`
  (`llmm/attention.mojo:1267`) already computes each row's running max `m`
  (post `attention_scale`, which matches K2's definition of `S_max`). Add an
  optional `head_max_ptr` (`[L, NH]` fp32) and an `atomic max` per row (as the
  usual int-bit-pattern trick for non-negative floats, or signed-compare on
  the int reinterpretation). `USE_FLASH_FWD` is `False`
  (`llmm/attention.mojo:1375`), so this is the only production path to hook;
  the flash kernel gets the same hook only if it is turned on later.
- **Multi-rank / ZeRO**: the optimizer shard is a flat element range
  (`optimizer_num_parameters`, `train_gpt2.mojo:2037`), which cuts matrices
  across ranks. Muon needs whole matrices. Phase 1 supports **WORLD_SIZE=1
  and ZeRO-0 (DDP, full grads on every rank, every rank runs the full Muon
  step redundantly)** and raises at startup for `zero_stage >= 1`. The
  sharded design (round-robin whole matrices across ranks, then allgather the
  updated rows, as modded-nanogpt does) is phase 3.
- **CPU**: NS needs a GEMM. The CPU path gets a plain fp32 reference NS
  (triple loop / existing `matmul_fwd` CPU kernel) so `make verify`-style CPU
  equivalence tests exist, but it is not a performance target. Keep the CPU
  dispatch comptime-guarded exactly like AdamW (bf16 builds crash AArch64
  codegen if a CPU target is instantiated; see memory note).

## 3. New code

```
llmm/muon.mojo                 MuonConfig, muon_momentum_kernel, newton_schulz (batched),
                               muon_apply_kernel, muon_update (dispatcher), qk_clip_apply
llmm/attention.mojo            optional per-head logit max in the causal softmax
train_gpt2.mojo                -opt / -mu / -qk_clip flags, Muon group table, update() branch,
                               setup-table + LLM-mode log keys (opt=, qk_max=)
llmm/checkpointing.mojo        optimizer_kind header slot
tests/test_muon.mojo           NS orthogonality + batched/transposed shapes (GPU, shared ctx)
tests/test_muon_equivalence.py Muon step vs a torch reference (same harness as test_adamw_equivalence.py)
tests/reference.py             torch Muon reference (NS5 in bf16, Moonlight scaling)
```

Kernels in `llmm/muon.mojo`:

1. `muon_momentum_kernel` (elementwise, batched over a group): reads
   `grad * grad_scale`, updates `M` in fp32, writes the Nesterov `U` as bf16
   into scratch `X[L, m, n]`.
2. Group norms: one `global_norm_squared_gpu` launch, then a tiny kernel that
   scales each slice of `X` by `1/(||U|| + eps)` (or fold the scale into the
   first NS GEMM's alpha).
3. `newton_schulz` (5 iterations, all strided-batched bf16 cuBLASLt):
   `A = X Xᵀ` (`[m,m]`), `B = b*A + c*A*A` (one GEMM with beta=b on a copy of
   A, or GEMM + axpy), `X = a*X + B*X`. When `m > n` run on `Xᵀ` (fc). The
   transpose is a layout choice, not a copy: `_matmul_cublaslt`'s
   `transA/transB` and `ld_*` parameters cover it.
4. `muon_apply_kernel` (elementwise): `W_master -= lr * (scale * O + wd * W)`
   with the shared master/bf16/SR store helper.
5. `qk_clip_apply` (elementwise over Q/K rows + bias of one layer, only when
   `head_max[l, h] > tau`): scale by `sqrt(tau / head_max)`.

Scratch (bf16, allocated once at 124M sizes): `X` for the largest group
(`fc`: `L·4C·C` = 56 MB), `A` and `B` (`L·C·C` = 14 MB each), plus `[L, NH]`
fp32 for head maxima. Negligible next to activations.

Cost estimate at 124M (`C=768`, `L=12`): the NS GEMMs total roughly
1.2 TFLOP per optimizer step. At a realistic 150–200 TFLOP/s bf16 on the GB10
that is ~6–8 ms against the current 134.6 ms step, i.e. ~5%. Muon's claimed
~1.35× fewer steps to a target loss dwarfs that, but the step-time delta must
be measured, not assumed (see §5).

## 4. Phases

**Phase 1 — Muon, single GPU, bf16 and fp32 builds.**
Kernels 1–4, the `-opt muon` / `-mu` flags, the update() branch, checkpoint
header slot, both tests. Exit: `tests/test_muon_equivalence.py` matches torch
to bf16 tolerance over a multi-step trajectory, `make test-mojo` green, and a
10-minute tinyshakespeare run shows loss going down with no NaNs.

**Phase 2 — QK-clip (`-qk_clip tau`, default 0 = off).**
Softmax hook, `[L, NH]` buffer, `qk_clip_apply`, `qk_max=` in verbose LLM-mode
logs. Exit: a unit test that plants a huge Q row and checks the post-step
logit max is ≤ `tau`; a normal run reports `qk_max` far below 100 and never
triggers.

**Phase 3 — Sharded Muon (ZeRO-1/2/3, multi-rank CPU harness).**
Matrix-granular ownership + allgather of updated rows. Only if the 8-GPU track
in `hyperparam_tuning_8gpu_2026-07-14.md` is revived; not needed for the GB10.

Out of scope, noted for later: NorMuon (per-row normalization), Polar Express
coefficients, freeing `v_memory` under Muon, Metal path (bf16 GEMM via MPS is
there, but untested for batched strides).

## 5. Validation and reporting

- Correctness: the two tests above, plus `make verify` unchanged (default
  optimizer is still AdamW, so every existing gate stays byte-identical).
- Convergence: GPT-2 124M on FineWeb, same `B·T`, same schedule as
  `gpt2_124m_fineweb_training_run.md`, AdamW vs Muon at matched step counts;
  report val loss at fixed steps and steps-to-3.28. Interleave runs in the
  same window (unified-memory tenant contention note) and check `nvidia-smi`
  first.
- Step time: `make benchmark-gpu` style timing, AdamW vs Muon, to put a real
  number on the NS overhead.
- Results land in a new `docs/ai/muon_training_run_<date>.md` following the
  full-run report shape; the parity table in the README is not touched.

## 6. Open decisions (defaults chosen, flag if you disagree)

1. Moonlight RMS scaling (shared LR) over modded-nanogpt's separate Muon LR.
2. Q/K/V orthogonalized as three `[C, C]` blocks, not one `[3C, C]` matrix.
3. Phase 1 raises for ZeRO stages ≥ 1 instead of silently falling back to
   AdamW.
4. Momentum `mu = 0.95`, Nesterov on, NS steps 5, eps 1e-7, `tau = 100`.

## AI use statement

Written with AI assistance (Claude Code, Fable 5.1 agent), directed by Evan
Owen.
