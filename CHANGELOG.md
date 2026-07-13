# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-13

### Fixed

- **Metal: T=1024 training-step regression from the batched attention scoreout.**
  The batched single-launch QKᵀ kernel wired into `_attention_bmm_scoreout` on
  2026-07-11 was gated only by `hd == 64 and T % 32 == 0`, so it also took over
  the full training sequence length T=1024. That kernel is a plain tiled SIMD
  GEMM and only wins while QKᵀ is dispatch-bound at short T; at T=1024 QKᵀ is
  compute-bound, where the per-head `linalg.matmul` path reaches far more of peak.
  The unconditional wire-in silently slowed the T=1024 step (M4 Max, B=4): bf16
  from ~499 to ~717 ms, fp32 from ~652 to ~881 ms, while PyTorch MPS was
  unchanged. Fix: gate the batched path to `T <= SCOREOUT_MAX_T` (256), so T=64
  keeps the batched win (164 vs 261 ms/step) and T=1024 takes the per-head
  fallback (502 vs 718 ms/step). The three gate constants (`SCOREOUT_HEAD_DIM`,
  `SCOREOUT_T_MULTIPLE`, `SCOREOUT_MAX_T`) are now named comptime values. Gated by
  the attention-equivalence battery (`tests/test_attention_equivalence.py`, 32
  cases).

### Performance

- **Apple Silicon (Metal) benchmark refreshed on an M4 Max (2026-07-13).** New
  six-arm figures at T=1024 and T=64 (llm.mojo fp32/bf16, PyTorch MPS fp32/bf16,
  MLX fp32/bf16). llm.mojo stays faster than PyTorch MPS at both lengths (bf16
  1.71x, fp32 1.25x at T=1024), but MLX leads on this machine at both lengths.
  The gap is the matmul: MAX's Metal `linalg.matmul` runs bf16 at only ~1.1x its
  own fp32 rate (no bf16 tensor cores), while MLX's steel_gemm gets ~2x, and a
  hand-written tensor-core GEMM does not compile on Metal in the pinned 1.0.0b3
  nightly (`TensorCore.store_d`). See `README.md` and `bench_gemm.mojo`.

## [Unreleased] - 2026-07-11

### Performance

- **Apple Silicon (Metal): bf16 training step −29%, now beats MLX bf16.** A
  profiling campaign on an M4 found the causal-attention QKᵀ path
  (`_attention_bmm_scoreout`) was issuing **BH=48 serial `linalg.matmul` launches
  per call** (one per head) on Metal — pure kernel-dispatch overhead that, at the
  benchmark's short T=64, made attention **53% of per-layer forward / 41% of
  per-layer backward**. A single-launch batched Metal kernel for this GEMM
  (`_attn_batched_scoreout_gpu`/`_launch_batched_scoreout`) already existed in
  `llmm/attention.mojo` but was dead code; wiring it into `_attention_bmm_scoreout`
  (gated `HAS_METAL and dtype == out_dtype`, with an `hd == 64 and T % 32 == 0`
  fast-path guard falling back to the per-head path for other shapes) cuts
  attention forward/backward ~−58%/−54% and the full bf16 step **~269 → ~190 ms**
  (same-window A/B). This moves llm.mojo bf16 from behind MLX bf16 (215.6 ms) to
  ahead of it; fp32 improves too (the batched path is not bf16-gated). The CUDA
  path is unchanged. Gated by the full `make test` battery (233 pytest + `test_gpt2`
  end-to-end equivalence, exercising the batched path at bf16 and fp32). See
  `docs/ai/metal_beat_mlx_campaign_2026-07-11.md`.
- **Metal: per-step allocation removed from grad-norm.** `calculate_grad_norm()`
  called `enqueue_create_buffer` + `enqueue_create_host_buffer` fresh every
  training step even though the reduction grid size is a runtime constant;
  persistent `grad_norm_out_buf`/`grad_norm_host_buf` fields (sized once in
  `allocate_optimizer_moments`) trim the update phase ~−3 ms. Optimizer math is
  bit-for-bit unchanged.
- **Metal bf16 GEMM lever documented as closed (no code change).** Confirmed via
  micro-benchmark + empirical `TensorCore` probes that a faster bf16 Metal GEMM is
  not expressible through Mojo's current Metal surface (`mma`/`store_d`
  unsupported; hand-tiled SIMD is 3–6× slower; bf16 `linalg.matmul` already ≈ its
  own fp32 via an internal simdgroup path inside MAX). Artifact: `bench_gemm.mojo`.

## [Unreleased] - 2026-07-10

### Added

- **FP8 mixed-precision training (Chunks A-G, `goal2/fp8-training`)** — GPT-2 124M now
  trains end-to-end with the four per-block linear GEMMs (QKV/attn-proj/MLP fc/MLP
  proj, forward + backward) in FP8 (E4M3 forward operands, E5M2 backward gradient
  operand, delayed per-tensor scaling with a 16-step amax history), while storage,
  LayerNorm, softmax, attention core, GELU, the LM head, and the AdamW optimizer stay
  bf16/fp32 exactly as in the existing mixed-precision build (`make build-fp8`).
  End-to-end verification (`docs/ai/fp8_training_design.md` Chunk F): the recalibrated
  148-tensor gradient gate (`make verify-fp8-grads`) passes (per-tensor cosine floor
  >0.93, relL2 envelope median 0.174/max 0.462 within the calibrated compounding
  bounds, zero depth-monotonicity violations, no NaN/Inf); two identical fp8 runs are
  bitwise-identical step-for-step (no scale/amax race); a 50-step fp8-vs-bf16 loss
  trajectory tracks within 0.6% median / 1.8% max relative delta with no drift. FP8 is
  currently **~36% slower** than bf16 at B=4,T=1024 on NVIDIA GB10 (quantize/amax
  kernel-launch overhead outweighs the small-batch GEMM saving at this scale) — a
  correct, gated, but not-yet-faster milestone; see
  `docs/ai/ai_assisted_optimizations_and_benchmarks.md` for the full gate results and
  perf breakdown, and `docs/ai/low_precision_gotchas.md` for implementation gotchas.

- **NVFP4 mixed-precision training (Chunks T1/T2a/T2b, `goal3b/fp4-training`)** — the
  MLP fc/fc_proj GEMMs of GPT-2 124M's middle transformer blocks (layers
  `[LLMM_FP4_FIRST, LLMM_FP4_LAST)`, default `[2, num_layer-2)`) now run in NVFP4
  (e2m1 data, two-level e4m3-block + fp32-tensor scaling) forward AND backward
  (`make build-fp4`), with the full recipe from `docs/ai/fp4_training_recipes_
  research.md`: RNE forward, stochastic rounding on the backward gradient operand,
  a Random Hadamard Transform on the weight-gradient GEMM (default on, `-D
  LLMM_FP4_NO_RHT=1` ablates it off), 2D 16x16 weight / 1D 1x16 activation-gradient
  block scaling. Everything else (qkv/attn-proj, attention, LayerNorm, GELU, the LM
  head, and the fp32 AdamW optimizer) stays bf16/fp32, unchanged. Closeout
  verification (`docs/ai/ai_assisted_optimizations_and_benchmarks.md`'s 2026-07-10
  FP4-closeout entry): two fp4 10-step runs against an exercised binary are
  bitwise-identical; a 50-step fp4-vs-bf16 loss envelope tracks within 0.89% median
  / 2.11% max relative delta with no drift; fp4 is currently **~36.5% slower** than
  bf16 at B=4,T=1024 on NVIDIA GB10 (the e2m1 GEMMs themselves are ~13% faster than
  bf16's, but a new quantize/amax/RHT-prep kernel family — 33% of total fp4 GPU
  time, split roughly evenly between NVFP4 quantization and the Wgrad RHT's
  transpose+Hadamard prep — outweighs that gain), matching fp8's "correct, gated,
  but not-yet-faster" milestone shape. A confirmed (3.2x, timing-comparison-based)
  strided-read coalescing pathology in `nvfp4_quantize_transpose` is documented and
  deferred (its packed-nibble/block-scale output makes a tile rewrite non-trivial,
  unlike fp8's analogous fix); a second, more mechanical instance in the RHT
  prep's naive transpose kernel had a ready fix implemented and fully gated, but
  was reverted after surfacing an unrelated, pre-existing, non-fp4-specific
  bit-stability fragility on a freshly-rebuilt binary's first invocation (root-
  caused as orthogonal to the fix itself, but left unshipped pending a follow-up
  session with `compute-sanitizer` available) — see
  `docs/ai/low_precision_gotchas.md` F1-F8 for the full gotcha trail.

## [Unreleased] - 2026-07-03

### Fixed

- **Training-correctness campaign: three target-independent backward-pass bugs** — A
  25-step fresh-data training run exposed rising validation loss, revealing that the
  original `test_gpt2` gradient checks (flat absolute tolerance 2.0) were vacuous for
  near-zero parameters and the 10-step loss trajectory was computed but never asserted.
  Three bugs were found and fixed, all predating the Metal port and affecting CUDA,
  GB10, and CPU equally:

  1. **Residual-skip gradient never seeded** (`llmm/layernorm.mojo:2122`,
     `train_gpt2.mojo:2304, 2424, 2440`) — `layernorm_fused_residual_bwd` only
     accumulates `+= LN_dinp`; the incoming residual-stream skip gradient
     `d(inp1+inp2)` was never pre-added. Result: block gradients decayed geometrically
     with depth (wpe grad ~10⁶× too small). Fixed by `residual_grad_broadcast`, called
     at three seed points in `backward()`.

  2. **GELU gradient fused into the FC matmul backward instead of PROJ**
     (`train_gpt2.mojo:2257, 2277`) — `use_gelu=True` was passed to the FC backward
     (below the GELU boundary) and `use_gelu=False` to the PROJ backward (which crosses
     it). The 4C-wide `fch` tensor was indexed with C-wide stride, corrupting `d_ln_2`
     (maxdiff 15.76 vs reference). Fixed by swapping `use_gelu` between the two MLP
     matmul backward calls.

  3. **Layer-0 LN1 backward passed the normed output instead of the pre-norm input**
     (`train_gpt2.mojo:2408–2410`) — `layernorm_bwd` needs the pre-norm input to form
     `xhat` and compute `dgamma`. Layer 0 passed `l_ln_1` (the normed output) instead
     of `acts.encoded` (the encoder output, the true LN1 input). `dgamma` was wrong;
     `dbeta` was unaffected. Fixed by passing `self.acts.encoded`.

- **Metal recompute-QKᵀ attention backward disabled** (`train_gpt2.mojo:1769, 2352`)
  — The Metal backward path that recomputed `QKᵀ` instead of reading stored softmax
  probs was found to amplify gradients ~1.4× per layer (compounding to ~460× error at
  the bottom layer in 12-layer GPT-2). Store-P re-enabled unconditionally for all
  targets pending a corrected recompute implementation (perf follow-up).

### Changed

- **`test_gpt2` hardened** (`test_gpt2.mojo`) — Mixed atol (0.01) + rtol (0.05 ×
  `ref_maxabs`) tolerance replaces flat absolute 2.0; L2 norm ratio guard (3×) catches
  dead gradients whose element-wise maxdiff is coincidentally small; 10-step loss
  trajectory now asserted at every step with tolerance 0.01 (matching llm.c). Expected
  losses fixture regenerated by
  `~/Workspace/scripts/llmm-metal-probes/gen_expected_losses.py` — the old fixture used
  β₂ 0.95 and weight_decay 0, not matching the test loop's actual update rule (β₂
  0.999, weight_decay 0.01).

### Performance

- **Official Metal benchmark post-correctness (2026-07-03 01:51, B=4, T=1024, M4 Max,
  cold GPU, 30s inter-arm cooldowns):**
  - llm.mojo fp32: **737.74 ms / 5552 tok/s**
  - llm.mojo bf16: **587.81 ms / 6968 tok/s** (fastest config)
  - PyTorch MPS fp32: 816.32 ms / 5018 tok/s
  - PyTorch MPS bf16: 845.67 ms / 4843 tok/s

  llm.mojo (Metal) beats PyTorch MPS on both precisions: **+10.6% faster in fp32,
  +30.6% faster in bf16**. bf16 is now the fastest Metal config (previously it
  appeared slower due to the broken recompute backward path; store-P re-enabled fixes
  this).

- **Metal kernel-optimization wave** (post-correctness): vectorized `bias_gelu`
  (2D-grid SIMD, 30 → 320-770 GB/s effective, −40 ms/step in both precisions) and
  a transposed-A register-tiled GEMM for the attention backward (eliminates 4
  rectangular transposes + 2 generic kvgrad kernels per layer, −51 ms/step bf16).
  Layernorm was investigated and confirmed structurally at its floor (five
  approaches measured and reverted). Final official 4-arm benchmark
  (B=4 T=1024, cold GPU, 30 s inter-arm cooldowns, 2026-07-03 10:48):
  - llm.mojo fp32: **652.06 ms / 6282 tok/s** (27% faster than PyTorch MPS fp32)
  - llm.mojo bf16: **498.92 ms / 8210 tok/s** (1.72× PyTorch MPS bf16; 7.3× the
    initial Metal port's 3627 ms)
  - PyTorch MPS fp32 / bf16: 830.27 / 857.26 ms

  The benchmark chart layout was also fixed (footer and legend now participate in
  constrained layout; no more overlapping text).
  Figure: `figures/benchmark_metal_b4_t1024_2026-07-03_1048_Apple-M4-Max_Mac-M4-Max.png`.

---

## [Unreleased] - 2026-07-01

### Fixed

- **CUDA forward regression from the Metal-support commit** (2026-07-02) — `f37cbd7`
  unconditionally de-fused the per-layer bias(+GELU) from the cuBLASLt GEMM epilogue
  for all targets (a workaround Metal needs), costing CUDA ~4 ms/step of forward and
  shifting bf16 numerics (step-0 loss 8.791504 → 8.792923), and leaving the
  gradient-checkpointing rematerialization on the old fused path. Fixed by moving the
  fused-vs-split decision into the kernel layer (`matmul_fwd` in `llmm/matmul.mojo`:
  fused epilogue on `HAS_CUBLAS`, split GEMM + `bias_gelu_fwd` elsewhere) and reverting
  the `train_gpt2.mojo` call sites to single un-branched calls (byte-identical to
  pre-Metal). CUDA losses bit-identical to the parity reference again, forward back to
  ~44–46 ms; portable, recompute (`LLMM_RECOMPUTE=1`), and CPU paths all verify green
  (CPU also regained its fused epilogue: ~105 vs ~170 ms/step). The regression was
  initially masked by a stale benchmark binary — validation now requires a
  freshness-checked build (see docs/ai/ai_assisted_optimizations_and_benchmarks.md).

### Performance

- **Merge kernel: coalesced-write + 8-wide vectorization** (2026-07-02) — Flipped `merge_fwd_gpu`'s thread→element mapping from head-layout-indexed (coalesced read, strided write) to token-layout-indexed with width=8 (coalesced 16-byte read *and* write from the same thread), landing as a new `merge_fwd_gpu_coalesced` kernel with host-side dispatch (`head_dim % 8 == 0`) alongside the always-correct width=1 fallback. ~14% faster in isolation (61.5 → 52.9 µs/call), bit-identical output. Found via a 4-agent parallel investigation (one real win, three rigorous dead ends — see the doc for all four). Gates: `make verify-gpu` green, full `make test` 235/235 including the odd-head_dim equivalence fixture.

- **GPU kernel alignment sweep** — Applied explicit `alignment=align_of[SIMD[dtype,width]]()` across backward-pass kernels (layernorm, attention, softmax, gelu, encoder) to match llm.c memory patterns. Win: ~1.5 ms per step (kernel-level measurement). All 16 gradient tensors verified.

- **Selective gradient zeroing** — Removed unnecessary 17.5 ms/step memset of the 3.29 GB `grad_acts_buf`. Replaced wholesale-fill with targeted 308 MB fills of accumulators only ({encoded, attn_proj, residual_2, fc_proj, residual_3}), exploiting the fused-residual-backward dataflow. Validated by fp32 equivalence checks, bf16 bit-identical loss trajectories, and poison-test (sentinel-fill) verification.

- **Layernorm backward broadcast fusion** — Fused the residual-gradient broadcast (`d_inp1 += dval; d_inp2 += dval`) into a single GPU kernel, eliminating the separate 2.4 ms broadcast sweep. Dataflow verified by enumerating all post-broadcast access sites (all writes, no consumption). Gates: fp32 verify, bf16 bit-identical trajectories.

- **Dead gradient-activation allocations removed** — Pruned `fused_classifier` and `logits` gradients never dereferenced on GPU (both cuBLAS and portable paths). Freed 1.9 GB from `grad_acts_buf` (3.29 GB → 1.37 GB), reducing footprint pressure on shared/unified-memory GPUs.

- **Profile-harness fairness fix** — `profile_gpt2.mojo` now runs `calculate_grad_norm` and gradient-scaled updates in the measurement loop, matching train_gpt2.mojo and llm.c behavior. Honest baseline: ~150.5 ms → **134.7 ms (parity with llm.c)**.

### Headline

- **🎯 Step-time parity reached with llm.c** — llm.mojo bf16 (B=4, T=1024, NVIDIA GB10) now matches llm.c step-for-step: **134.6 ms (mojo) vs 134.7 ms (llm.c)**, equal in quiet-window interleaved A/B testing. Cumulative optimization: fp32 baseline (2612 ms, June 30) → bf16 parity (134.7 ms, July 1) = **~19.4× speedup**. Full training-fidelity constraints maintained: fp32 optimizer state, bf16 loss bit-stability, and working non-CUDA GPU fallback (`LLMM_FORCE_PORTABLE_GPU`).

- **Ahead of llm.c, 3 repeated runs (2026-07-02)** — after the merge kernel fix, on a quiet GPU: llm.mojo **135.15–136.10 ms** vs llm.c **136.49–140.46 ms**, consistently at or ahead across all three, with tighter variance (std 0.9–1.1 vs 1.1–4.9).

### Changed

- **Non-CUDA GPU compatibility** — Introduced `HAS_CUBLAS` and `LLMM_FORCE_PORTABLE_GPU` flag to maintain vendor-neutral GPU paths. cuBLAS fast paths (classifier, linear GEMMs) now fall back to MAX `linalg.matmul` on non-NVIDIA; attention falls back to pure-Mojo flash algorithm. Portable mode tested and green (e.g., 5.3557653 loss = within noise).

- **Test fix** — Updated `tests/test_zero_equivalence.mojo` to reflect gradient allocation changes and selective-zeroing behavior.

### Technical Details

- Session entries ㉘–㉝ in `docs/ai/ai_assisted_optimizations_and_benchmarks.md` document the full campaign: root-cause profiling, A/B measurements, dead-end hypotheses, and the unified-memory pressure diagnosis that explained the earlier 1.36× GEMM slowdown.
- The 2026-07-02 sections (softmax re-investigation + four-agent parallel investigation) document the merge kernel win above plus three dead ends with mechanism: cuBLASLt workspace-size tuning (already matched llm.c, no effect at any budget), softmax fast-exp/cache-eviction hints (statistical ties under locked-clock ncu), and layernorm dgamma/dbeta fusion + the entire async-copy/latency-hiding lever family (GB10's memory-bound kernels proven bandwidth-bound, not latency-bound). Scratch prototypes for all four kept in-tree with RESULT-header summaries.

---

## [0.1.0] - 2026-06-30

### Added

- Initial Mojo port of Karpathy's llm.c (GPT-2 124M training).
- Full transformer training loop: forward pass, backpropagation, and AdamW optimizer.
- GPU kernel implementations: matmul, attention (flash algorithm), softmax, cross-entropy, layer normalization, GELU, embeddings.
- CPU equivalence tests for all major operations.
- Distributed data loading with sharding support.
- Mixed-precision training scaffolding (fp32/bf16 storage dtypes; bf16 mixed
  precision reached full parity 2026-07-01, see the `[Unreleased] - 2026-07-01`
  entry below — fp8 was not yet functional at this 0.1.0 tag, and fp16 was never
  implemented as a distinct dtype; see the `[Unreleased] - 2026-07-10` entry above
  for FP8's actual functional/gated debut).
- Model checkpointing and weight export.
- Benchmark harness for profiling against llm.c and PyTorch.
- Support for both CUDA and portable GPU execution paths.
- Gradient checkpointing and zero-redundancy optimizer (ZeRO) stages 1–3.

### Performance Notes

- Achieved stable training with bf16 loss trajectories matching PyTorch reference.
- Initial performance baseline: ~2612 ms/step (fp32) on NVIDIA GB10.
- GPU profile harness with per-kernel breakdown via nvidia-smi and ncu metrics.

### Documentation

- Comprehensive profiling and optimization guide in `docs/ai/ai_assisted_optimizations_and_benchmarks.md`.
- README with setup, data preparation, and training instructions.
- AI use statement documenting Claude Code agent roles in the optimization campaign.

---

## Historical Summary (2026-05-31 to 2026-06-30)

### June 2026 Kernel Development

- **Transformer architecture**: embeddings (wte, wpe), causal self-attention with KV-cache, residual blocks, layer norm, MLP (with GELU activation).
- **Backpropagation**: full gradient flow for all operations, including fused-residual backward, attention softmax backward, cross-entropy gradient.
- **Optimizer**: AdamW with weight decay scheduling, learning rate warmup, and gradient clipping.
- **Data pipeline**: distributed data loader with per-rank sharding, support for .bin format (Karpathy's format).

### Beta Compatibility

- Migrated codebase to Mojo 1.0.0b3 during development.
- Fixed syntax and import changes in new beta version.

### Early Profiling and Testing

- Initial bf16 mixed-precision support (June 30).
- Gradient accumulation and distributed data parallel (DDP) scaffolding.
- Per-operation equivalence tests against PyTorch reference.

---

*For details on the optimization campaign (entries ㉘–㉝ and parity diagnosis), see `docs/ai/ai_assisted_optimizations_and_benchmarks.md`.*
