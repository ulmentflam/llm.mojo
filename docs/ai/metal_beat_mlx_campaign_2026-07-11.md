# Metal campaign — beat MLX bf16 on Apple Silicon (2026-07-11)

**Machine:** Apple M4 (base, 10 GPU cores), macOS 26.5, Mojo 1.0.0b3.
**Config:** B=4, T=64 (the `make benchmark-metal` config on this box).
**Goal:** llm.mojo bf16 currently loses to MLX bf16; close the gap and win.

## Baseline standings (figures/benchmark_metal_b4_t64_2026-07-11_Apple-M4_Mac-M4.json)

| arm | ms/step | tok/s |
|---|---:|---:|
| MLX bf16 | **215.6** | 1187 |
| llm.mojo bf16 | 270.7 | 946 |
| llm.mojo fp32 | 283.6 | 903 |
| MLX fp32 | 299.0 | 856 |
| PyTorch MPS fp32 | 484.6 | 528 |
| PyTorch MPS bf16 | 602.5 | 425 |

We **win fp32** (283.6 vs 299.0) but **lose bf16** (270.7 vs 215.6, MLX 1.26× ahead).
Target: get llm.mojo bf16 below ~215 ms/step.

## Phase decomposition (profile_gpt2_bf16, 12 layers, B=4 T=64, steady-state)

Measured on-box via `build/profile_gpt2_bf16` (forward/backward/update each
`ctx.synchronize()`-bracketed). Total ≈ 269 ms, matching the benchmark.

| phase | ms | share |
|---|---:|---:|
| forward | ~100 | 37% |
| backward | ~128 | 47% |
| update (AdamW + grad-norm) | ~41 | 15% |

Layer-count differential (1 vs 6 vs 12 layers) splits fixed vs per-layer:

| | forward | backward | total |
|---|---:|---:|---:|
| per-layer (×12 transformer blocks) | 7.5×12 = 90 | 8.6×12 = 103 | **193 (72%)** |
| fixed (encoder + classifier + ln_f) | 9.7 | 22.9 | 32.6 (12%) |
| update | — | — | 41 (15%) |

## Root-cause hypothesis (the crux)

**bf16 is only 1.05× faster than fp32 for us (283.6→270.7), but MLX gets 1.39×
from bf16 (299.0→215.6).** Our bf16 matmul is *not* getting real bf16 throughput
— `llmm/matmul.mojo` has **zero** `simdgroup` uses; the Metal GEMM path routes
through stdlib `linalg.matmul[target=...]`, which does not appear to hit Apple's
`simdgroup_matrix` bf16 tensor path. MLX's hand-tuned `steel_matmul` does.

Ranked levers:
1. **bf16 Metal GEMM** (matmul.mojo) — biggest: per-layer is 72% of the step and
   dominated by 4 GEMMs/layer that today run at ~fp32 speed. → workstream `metal-gemm`.
2. **AdamW + grad-norm update** (adamw.mojo, global_norm.mojo) — 41 ms / 15%,
   fully independent. → workstream `metal-adamw`.
3. **Attention / softmax** (attention.mojo, softmax.mojo) — part of per-layer;
   MLX uses fused SDPA. → workstream `metal-attn`.

## Measurement protocol for this campaign

- The M4 GPU throttles after ~8 s of sustained load and there is only one GPU,
  so **concurrent timing runs are unreliable**. Per-workstream timings are
  *indicative*; the orchestrator re-measures winners serially before merging.
- Per-workstream correctness gate (cheap, NOT full `make test`): (a) `make
  format lint` green, (b) the changed kernel's output matches the baseline
  kernel numerically on random input (drop-in replacement), (c) the profile
  harness loss trajectory tracks the baseline (loss 9.62→~0.68 over 12 steps at
  L=12 B=4 T=64). Full `make test` + `make check` run once at integration.

## Results log

### metal-attn (attention/softmax) — BIG WIN: −29% step (269→191ms), beats MLX

Measure-first found attention was **NOT** small at T=64: attn_fwd ≈4.94ms/layer
(**53%** of per-layer forward), attn_bwd ≈5.00ms/layer (**41%** of per-layer
backward). Root cause: `_attention_bmm_scoreout` (QKᵀ fwd, dOᵀ·V/QKᵀ-recompute
bwd) fell through to a **Python-level loop over BH=48 heads = 48 serial
`linalg.matmul` launches** — pure dispatch overhead at T=64 (~25M FLOP, tiny).
Its sibling ops (`_attention_bmm_headout`, `_attention_bmm_kvgrad`) already have
single-launch batched Metal kernels; a matching one for scoreout
(`_attn_batched_scoreout_gpu`/`_launch_batched_scoreout`) **was already fully
implemented in the file but never wired up (dead code)**.

**Fix (25 lines, `llmm/attention.mojo` only):** wire `_launch_batched_scoreout`
into `_attention_bmm_scoreout`, gated `HAS_METAL and dtype == out_dtype`, runtime
guard `hd == 64 and T % 32 == 0` (training's T=64/T=1024 qualify; odd T from
generation/tests falls back to the per-head path). CUDA path untouched.

Thermally-matched A/B: attn_fwd 4.80→2.00ms/layer (−58%), attn_bwd
4.92→2.26ms/layer (−54%). Clean full-step: fwd 100.2→64.4ms (−36%), bwd
127.5→87.6ms (−31%), **total ≈269→191ms (−29%)** — beats MLX bf16 215ms. Loss
matched baseline (9.607/0.916/0.686). `make format lint` green. Patch saved;
commit blocked by 1Password signing in sandbox (integrate via patch).
**Pending: serial re-measure on idle GPU + `make test` (the batched scoreout
kernel was previously dead code — must confirm no latent reason).**

### metal-gemm (bf16 GEMM) — LEVER CLOSED in Mojo (negative result, no change)

Micro-bench (M4, one process; ratios reliable, absolutes inflated by concurrent
agent GPU load). **bf16 `linalg.matmul` is already ≈ its own fp32** (1.4–2.0
TFLOP/s) — it uses an internal simdgroup path but yields no true bf16 throughput,
and that is inside MAX, not our code. Every Mojo-expressible candidate loses:
- Mojo's only simdgroup-matrix API is `layout.tensor_core.TensorCore`. On this
  Metal target (b3.dev2026062706): bf16 `mma` 16×8×16 → "target does not support
  operation: mma"; 8×8×8 / 8×8×4 → `load_a`/`load_b`/`mma_op` compile but
  **`store_d` fails to instantiate** for every out-dtype/address-space (can run
  the multiply, can't retrieve the accumulator). Tiled tensor-core bf16 GEMM is
  **not expressible**. (The `USE_FLASH_FWD=False` TensorCore kernel in
  attention.mojo is dead code, never Metal-compiled.)
- Hand register-tiled SIMD GEMM (fp32-accumulate, 2 tile configs): **3–6× slower**
  — Apple GPUs accelerate bf16 only via `simdgroup_matrix`; scalar bf16 runs at
  the fp32 ALU rate, far below linalg's internal simdgroup throughput. Output
  matched linalg bf16 with max_rel=0.0.

**Verdict:** the bf16-GEMM lever is closed through Mojo's current Metal surface; a
real win would need Modular to fix `TensorCore.store_d`+bf16-simdgroup on Metal,
or a raw-MSL `simdgroup_matrix<bfloat>` custom op via FFI (out of scope, "not in
Mojo"). Artifact: `bench_gemm.mojo` (records the full TensorCore probe matrix).
The campaign win therefore comes entirely from the non-GEMM levers below.

### metal-adamw (update phase) — small win: persistent grad-norm buffers (~3ms)

Split the ~41ms update: `adamw_update` ≈35-37ms, `calculate_grad_norm` ≈4.1-4.9ms.
**AdamW kernel: no change** — already a single fused vectorized pass at ~97GB/s
(~81% of M4's ~120GB/s peak); swept width{4,8,16}×BLOCK{128,256,512}, all in the
same band; 28 bytes/elem is the floor for fp32-master semantics. **Win in
`calculate_grad_norm()`:** it called `enqueue_create_buffer` + `enqueue_create_host_buffer`
*fresh every step* though `grid_x` is a runtime constant. Fix: persistent
`grad_norm_out_buf`/`grad_norm_host_buf`/`grad_norm_grid_x` fields on `GPT2`,
allocated once in `allocate_optimizer_moments()` (mirrors `m_buf`/`v_buf`
convention), reused each step. grad_norm 4.5→2.8ms; update phase ~41→~38ms. Loss
**bit-for-bit identical**. `make format lint` green. File: `train_gpt2.mojo` only
(adamw.mojo/global_norm.mojo untouched). Patch saved.

### metal-clsenc (classifier + encoder) — NOT MATERIAL, no change

Measured (L=1, bf16, steady-state, temp `ctx.synchronize()`+timers, reverted):
`fused_classifier` (fwd+bwd, single fused launch) ≈3.1ms; `encoder_fwd` ≈0.30ms;
`encoder_bwd` (bucketed atomic-free scatter) ≈0.27ms. **Total ≈3.7ms**, under the
6ms materiality bar. These paths are already near-optimal: 2-pass online
softmax+CE+grad (the numerically-stable minimum), MIN_FINITE max-init, 128-bit
vectorized loads, bucketed scatter-add avoiding atomics. **No change.** The rest
of the ~32.6ms fixed cost is `ln_f` (already-fused LN backward) + the lm_head/wte
GEMM backward (~20ms) — the latter is `matmul.mojo`, so the GEMM workstream
covers it. Loss trajectory matched baseline exactly (9.623/0.916/0.681).

## Integration & verification

**Landed changes** (2 files, both cleanly composable — disjoint):
- `llmm/attention.mojo`: wire `_launch_batched_scoreout` into `_attention_bmm_scoreout`
  (the metal-attn win — the −29% step-time driver).
- `train_gpt2.mojo`: persistent grad-norm buffers (the metal-adamw win).
- `bench_gemm.mojo`: added as a research artifact (records the closed-GEMM-lever
  probe matrix; not in the build/lint paths).

**Gate battery (all green, run on the assembled tree):**
- `make format` — clean (48+66 files unchanged).
- `make lint` — 0 errors.
- `make check` — lint + build train (fp32) + build profile (fp32) all compile;
  `make build-profile-bf16` also compiles.
- `make test` — **233 passed, 15 skipped** (pytest equivalence suites) + `test-mojo`
  all suites 0-failed, including `test_gpt2` end-to-end (activations + gradients +
  loss vs `gpt2_124M_debug_state.bin`, which runs bf16 **and** fp32 at the exact
  B=4/T=64/hd=64 shape that exercises the newly-activated batched scoreout path).
  This clears the one risk flagged for the attention change (it activated a kernel
  that had been dead code): no latent correctness reason existed.

**Performance — the trustworthy signals** (see thermal caveat below):
- metal-attn thermally-matched A/B (revert→build→measure→restore→build→measure in
  one window, robust to ambient thermal): attn_fwd −58%, attn_bwd −54%; full step
  **269 → 191 ms (−29%)**.
- Assembled-tree profile (`profile_gpt2_bf16`, L=12/B=4/T=64, same steps-6-11
  methodology that reproduced the 270.68ms benchmark at baseline): **bf16 269 →
  ~190 ms**. fp32 early (pre-throttle) steps ≈214–224 ms vs 283.6 baseline — the
  batched scoreout is gated `dtype==out_dtype` (not bf16-only), so **fp32 improves
  too**.
- Both put llm.mojo bf16 **below MLX bf16 (215.6 ms)** — the campaign goal.
- **Orchestrator self-verified A/B (independent of the sub-agent report).** Same
  window, back-to-back, pre-throttle steps 1–4 (so thermal *and* the concurrent
  iCloud-sync CPU contention cancel in the delta), attention change stashed/restored
  with adamw held constant in both: **integrated 192.6 ms vs attention-reverted
  259.1 ms → −66.5 ms (−25.7%)** from the batched scoreout. Integrated **192.6 ms <
  MLX 215.6 ms**. Because every remaining error source (throttle, iCloud CPU load)
  only *inflates* a measurement, 192.6 ms is an **upper bound** on the true
  cold/idle number — the win holds a fortiori.

**Thermal caveat on absolute benchmarks.** This box is a *fanless* M4 **Air**.
After the full campaign (4 concurrent agents + measurements) it was thermally
saturated; a `make benchmark-metal` run in that state returned every arm 2–3×
inflated with huge variance (llm.mojo bf16 median 558 std 397; even MLX bf16 460
vs its 215 reference) — invalid, discarded. The official figure/JSON must be
regenerated from a **cold** machine (30s inter-arm cooldowns assume a cold start;
they cannot recover a deeply-saturated passive-cooled Air). The relative deltas
above are thermal-robust (same-window A/B); the absolute headline number is
pending a cold-start `make benchmark-metal`.
