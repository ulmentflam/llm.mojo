# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Mixed-precision training (fp32, bf16, fp16, fp8).
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
