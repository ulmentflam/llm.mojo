# Low-precision training (TF32/FP8/FP4) campaigns: gotchas and changes

A consolidated technical log of the TF32 (goal1/fp32-parity), FP8 (goal2/fp8-training),
and FP4 (goal3/fp4-research) campaigns to extend `llm.mojo`'s training path to mixed-precision
regimes on NVIDIA GB10 (Grace-Blackwell, sm_121, aarch64). This document captures gotchas,
out-of-the-ordinary implementation changes, and non-obvious updates that future engineers
or agents extending low-precision support must not rediscover the hard way.

**Note at the top:** Complete audit of TF32/FP8/FP4 campaigns (2026-07-10, campaigns consolidated).
All three precision paths are integrated and gated. This document captures gotchas across the full campaign
lifecycle for future reference and extension.

**Related documentation:**
- The FP8 design document (`fp8_training_design.md` on this branch) covers the scheme,
  the dtype-generic layer (`llmm/lowp.mojo`), precision axis, and integration points.
- The FP4 research documents (`fp4_modular_support_research.md`, `fp4_training_recipes_research.md`
  on goal3/fp4-research branch) cover hardware/toolchain feasibility and recipe design.
- The NVIDIA optimization campaign (`ai_assisted_optimizations_and_benchmarks.md` on goal1/fp32-parity)
  documents the TF32 correctness and performance story.

**Validation gate:** "test passes" throughout this document means `make test` (`test_gpt2 gpu` path)
runs green. That is the ground-truth correctness gate for all GPU precision changes.

---

## TOOLCHAIN (Mojo 1.0.0b3.dev2026062706 / MAX 26.5 nightly)

### T1 — Mojo pop.cast for fp8 dtypes is BROKEN on GPU targets

**What breaks:** Mojo's generic fp8 dtype support (e4m3fn, e5m2, their unsigned variants)
compiles on the host (CPU target) with full SIMD arithmetic working correctly. On GPU targets,
any `pop.cast` from fp8 to higher precision (fp32, bf16) that feeds arithmetic fails lowering:
the operation is not implemented in the GPU-target backend.

**Exact error:** When GPU code calls `.cast[float32]()` on an fp8 value and uses the result
in arithmetic (not as a bare passthrough):
```
error: conversion from 'f8e4m3fn' to 'f32' is not implemented
note: see current operation: %24 = "pop.cast"(%23) : (!kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32>
```

**Effect:** Elementwise fp8 quantize/dequantize kernels (the standard pattern `load fp8 ->
cast to bf16 -> compute -> cast to fp8 -> store`) do not compile on GPU. MAX's own generic
`linalg.matmul` dispatcher, which internally performs fp8→f32 upcasts, also fails with the
same error when targeting GPU. Host-side fp8 casts and arithmetic work perfectly (Probe 1).

**Workaround shipped:** Mojo's `pop.cast` for fp8→higher-precision on GPU is unimplemented
in this toolchain (1.0.0b3.dev2026062706). **The working FP8 GEMM path consumes raw fp8
register bit patterns directly via vendor libraries** (cuBLASLt) or manual bit-manipulation
encoders on uint8-viewed buffers, **never** through Mojo's generic `pop.cast`. Host-side fp8
casts do work and are used for test references and host-side quantization. For any fp8
computation that requires the missing GPU-target cast, use host-side casts (validated with
fp32 references) as the fallback.

**Source:** `tests/probe_fp8/RESULTS.md` (this branch), Probes 1–3; verified on Mojo 1.0.0b3.dev2026062706.

### T2 — fp8 has NO AArch64 cpu-target codegen hazard

**Finding:** bf16 on this toolchain crashes AArch64 codegen if any `"cpu"`-target dtype
instantiation happens (landmine #1 in fp8_training_design.md). fp8 does **not** have this
hazard — fp8 CPU-target code (plain `vectorize()` loops, comptime `target="cpu"` dispatch)
builds clean on AArch64.

**Why:** The fp8→fp32 cast (Probe 1) *is* implemented for the CPU target; only the GPU
target lowering lacks it (T1). The bf16 crash is a separate, bf16-specific AArch64 backend bug.

**Implication:** fp8 CPU-side code does not need a comptime gpu-only guard purely to avoid
a codegen crash, **but** CPU-side fp8 kernels are moot for training anyway (Probe 2 shows GPU
elementwise compute is broken), so the guard for "no low-precision on CPU" remains for
architectural consistency.

**Source:** `tests/probe_fp8/RESULTS.md` (Probe 5); verified on GB10/AArch64.

### T3 — This toolchain rejects `fn` for public functions (use `def`)

**Gotcha:** Mojo 1.0.0b3 deprecated `fn` (the old keyword) in favor of `def` for public
function declarations. The `build-fp8` target and test harnesses expect `def`. Any new
low-precision helper functions must use `def`.

### T4 — This toolchain has no `constrained` keyword (use `comptime assert`)

**Gotcha:** Old-style Mojo `fn` parameter constraints (`fn foo[T: DType]` where `T` is constrained
to a specific set) are no longer supported; use `comptime assert` inside the function body to
validate type parameters at comptime.

### T5 — Parallel `make` invocations backgrounded with & in one shell are flaky

**Gotcha:** Running `make target1 & make target2 & wait` in a single shell invocation produces
spurious "No rule to make target" errors due to make's parallel jobserver interfering across
invocations. **Build sequentially:** `make target1 && make target2`.

---

## cuBLASLt (12.9.2.10)

### C1 — FP8 GEMM: e4m3×e4m3→bf16 WORKS; e4m3 output NOT SUPPORTED

**Working path:** e4m3 (fp8) A operand × e4m3 B operand → bf16 (or fp32) accumulate/output,
TN layout (transA=True, transB=False), K a multiple of 16. This is the standard Transformer-Engine
HYBRID pattern: low-precision GEMM operands, high-precision output.

**Not working:** e4m3×e4m3→e4m3 (fp8 output) returns `CUBLAS_STATUS_NOT_SUPPORTED` at runtime.
Likely requires an explicit `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` attribute (not explored
further as it falls outside the lowest-risk FP8 path).

**Design implication:** Storage stays bf16, fp8 is transient GEMM-operand only. The GEMM
writes bf16 output back into the bf16 activation/gradient pipeline. This choice sidesteps
the fp8-output scaling complexity and keeps the host↔device buffer element sizes unchanged
(avoiding landmine #2: the host-buffer element-size mismatch that caused the bf16 NaN bug).

**Source:** `tests/probe_fp8/RESULTS.md` (Probes 4a–4b); verified on GB10/sm_121 with
cuBLASLt 12.9.2.10.

### C2 — NVFP4: sm_120 block-scaled kernels dispatch unmodified on sm_121

**Finding:** cuBLASLt 12.9.2.10 (the one pinned in `.pixi`) ships sm_120 NVFP4 block-scaled
GEMM kernels (`cutlass3x_sm120_bstensorop_…_ue4m3xe2m1_…_vs16`, block=16, e2m1 data, e4m3 scales).
These are *named* sm_120 cubins. Yet the **sm_120-named cubin executes unmodified on this box's
sm_121 hardware** — no `CUBLAS_STATUS_ARCH_MISMATCH`, no silent CPU fallback, no driver update
required.

**Probe result:** Direct execution of FP4 `cublasLtMatmul` with block-scale descriptors returns
`CUBLAS_STATUS_SUCCESS`, nsys confirms the sm_120 kernel ran, and numerics match (rel L2 = 0.1445,
matching the software dequant reference to 4 decimal places).

**Implication:** cuBLASLt FP4 interop is viable on GB10 via the `_matmul_cublaslt` FFI bindings
(which llm.mojo already uses for bf16). No hand-written Mojo FP4 kernel is needed for dispatch;
the installed vendor library is sufficient.

**Source:** `tests/probe_fp4/RESULTS.md` (goal3/fp4-research branch); verified on GB10/sm_121
with cuBLASLt 12.9.2.10.

### C3 — NVFP4 conventions: e2m1 packing and block-scale swizzle must match PyTorch exactly

**Gotcha:** NVFP4 e2m1 nibble packing (2 values per byte) and block-scale swizzle (the
128×4-tile / 32×4×4-internal cuBLAS layout) have specific bit-for-bit conventions. The fp4
probe's quantization code uses:
- **e2m1 data:** OCP MX 4-bit table `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` × sign; nearest-value encode;
  2 values/byte, even index → low nibble, odd → high nibble (convention cross-checked against
  PyTorch's `pack_uint4`).
- **e4m3 block scale:** standard FP8 E4M3, 1 scale per 16-element block along K, derived per
  the cuBLAS doc §3.1.4.3.2 formula (and against PyTorch's `to_blocked()`/`from_blocked()`).

**Comparison bug:** The fp4 probe's first-pass comparison code had a column-major-vs-row-major
output indexing bug (cuBLASLt writes D column-major) that produced a spurious ~1.4 rel-L2 error
on *both* FP4 and bf16 control arms identically — making it look like a kernel bug when the
fault was in the readback indexing. **Always run a known-good control arm through new comparison
harnesses.** A correct bf16 control matched the reference to rel L2 = 0.0029, confirming the
layout scheme itself was right.

**Source:** `tests/probe_fp4/RESULTS.md` (goal3/fp4-research branch), "Numeric check" and
first-pass-version note.

### C4 — cuBLASLt's VEC16_UE4M3 scale mode is single-level; NVFP4's two-level scale needs a
post-GEMM correction the vendor call does NOT apply for you

**Finding:** `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` only consumes the e4m3 CODE from the
scale-pointer buffer and applies `decode_e4m3(code)` per 16-element block *inside* the GEMM —
it has **no notion of a second, per-tensor fp32 multiplier**. NVIDIA's own NVFP4 recipe (and
`llmm/nvfp4_quant.mojo`'s `nvfp4_quantize`) is a genuinely two-level scheme
(`original ≈ e2m1_value * decode_e4m3(block_code) * tensor_scale`, so the block scale is
*already* pre-divided by `tensor_scale` before being narrowed to e4m3 — see `nvfp4_quant.mojo`'s
module docstring). Feed that scale buffer straight to `A/B_SCALE_POINTER` and cuBLASLt's raw
output is `D_true / (tensor_scale_A * tensor_scale_B)`, not `D_true`.

**Symptom:** A first-pass FP4 GEMM (this chunk) that omitted the correction produced rel-L2 in
the 1e2–1e13 range (not a modest error — `tensor_scale` for gaussian data is a small fraction,
so `1/tensor_scale` blows the result up by orders of magnitude) while a same-harness bf16
control arm through the *identical* cuBLASLt call path landed fine (~0.0017 rel-L2) — proof the
bug was in the FP4-specific scale handling, not the GEMM dispatch/layout/readback (that class of
bug is C3's "always run a control arm" lesson, generalized: a broken harness makes *both* arms
wrong identically; a broken FP4-specific step makes only the FP4 arm wrong while the control
stays correct).

**Fix:** Apply a small elementwise post-GEMM kernel multiplying the raw `bf16` D buffer by
`tensor_scale_A * tensor_scale_B` (both are device-resident fp32 scalars, no host readback
needed — matches the codebase's existing "no host sync for scale state" principle). See
`llmm/matmul.mojo`'s `_nvfp4_post_scale_gpu`/`lowp_gemm_fp4`.

**Caveat this introduces:** the post-scale multiplies the *entire* `D` buffer, so it is only
correct when the GEMM wrote a fresh `D` (`beta=0`). A `beta=1` accumulate (needed for Wgrad
grad-accumulation across micro-steps) would incorrectly rescale the pre-existing accumulated
value too. `lowp_gemm_fp4` currently raises rather than silently mis-accumulate; supporting
`accumulate=True` correctly needs an extra raw-output scratch buffer (post-scale-then-add
instead of post-scale-in-place) — deferred to whichever chunk actually wires Wgrad
accumulation.

**Source:** `tests/test_lowp_gemm_fp4.mojo` (goal3b/fp4-training branch, FP4-GEMM chunk),
gaussian-512/MLP-shape test cases; `llmm/matmul.mojo`'s `_nvfp4_post_scale_gpu` module comment.

### C5 — NVFP4 "2D 16x16" weight block scaling is a *shared value*, not a *shrunk buffer*

**Finding:** cuBLASLt's block-scale layout is inherently one e4m3 scale per **physical row**
per 16-element K-block ("the scaling factor is stored for each 16-element block in the
innermost dimension of the corresponding data tensor" — `cublasLtMatmulMatrixScale_t`
docstring). There is no native cuBLASLt concept of a physically-compressed `rows/16`-row scale
tensor. The recipe's "2D 16x16" weight scaling (one shared scale per 16-row × 16-column tile,
so the same weight quantizes identically read row-major in Fprop or column-major in Dgrad) has
to be realized as: compute one e4m3 value per 16×16 tile, then **replicate that same code
across all 16 physical scale-buffer rows the tile covers** — the buffer's row extent must
always equal the data tensor's actual row count, never `rows/16`.

**Symptom:** An earlier version of `llmm/nvfp4_quant.mojo`'s `nvfp4_scale_buffer_size`/
`_nvfp4_quantize_gpu` sized/wrote the 2D-scaled (`BLOCK_ROWS=16`) scale buffer at `rows/16`
physical rows — self-consistent with (and therefore invisible to) its own
`nvfp4_dequant_reference`, since both sides of that pure-software round-trip agreed on the same
(wrong) convention. It only surfaced once the buffer was fed to the *real* cuBLASLt GEMM (which
reads swizzle offsets assuming the full `rows` row count from the operand's own matrix layout,
not from anything `nvfp4_quantize` told it): cuBLASLt read past the undersized buffer for
weight rows beyond the first `rows/16` and returned `NaN`s for the affected output elements
(GPT-2 MLP-shape test, `b_block_rows=16`, N=3072 → NaN at the 214th weight row; N=768 → NaN at
row 48). A pure quantize→dequant roundtrip test (`tests/test_nvfp4_quant.mojo`) could not have
caught this class of bug — it exercises the same (buggy) convention on both ends of the
comparison and can never disagree with itself.

**Lesson (generalizes C3's control-arm lesson):** a self-consistent software round-trip only
proves the quantize/dequant pair agree with *each other*; it proves nothing about whether they
agree with the actual consuming hardware API's own layout contract. Only feeding the real
buffer to the real vendor call (or a byte-for-byte contract test against the vendor's
documented layout) exercises that.

**Fix:** `nvfp4_scale_buffer_size(rows, k, BLOCK_ROWS)` now always sizes for `rows` physical
rows regardless of `BLOCK_ROWS`; `_nvfp4_quantize_gpu` computes one scale value per
`BLOCK_ROWS`-row tile as before but writes it to all `BLOCK_ROWS` physical row offsets;
`nvfp4_dequant_reference` reads from a tile's first physical row (all rows in the tile carry
the same code). `BLOCK_ROWS=1` (activations/gradients) is numerically a no-op change (a
1-row "tile" already behaved this way).

**Source:** `tests/test_lowp_gemm_fp4.mojo` (goal3b/fp4-training branch, FP4-GEMM chunk),
`test_fp4_gemm_mlp_fc_up_shape`/`test_fp4_gemm_mlp_fc_down_shape`; fix in
`llmm/nvfp4_quant.mojo`'s `nvfp4_scale_buffer_size`/`_nvfp4_quantize_gpu`/
`nvfp4_dequant_reference`.

---

## TF32 / fp32 (goal1/fp32-parity branch)

### TF1 — "fp32 parity with llm.c" means TF32-vs-TF32, not IEEE-vs-IEEE

**Baseline:** llm.c enables TF32 on NVIDIA sm≥80 via `CUBLAS_COMPUTE_32F_FAST_TF32` by default
(llm.c `train_gpt2.cu:1614–18`). Its "fp32" parity is actually **TF32 computation**, trading
IEEE-754 precision for tensor-core dispatch on Tensor Cores.

**The bug:** llm.mojo's `_matmul_cublaslt` in `llmm/matmul.mojo` hardcoded `ComputeType.COMPUTE_32F`
(IEEE fp32 GEMM), not `COMPUTE_32F_FAST_TF32`. This silently disabled tensor cores for fp32,
causing a **1.47× step-time regression** (measured 416.7→282.8 ms/step after fix) vs llm.c.

**Non-obvious:** The same hardcoded compute type was harmless in bf16 and invisible in review
because bf16 input dtype alone triggers tensor cores; TF32 is purely a compute-precision flag
that only matters for fp32 inputs.

**Fix:** Goal1 implements a comptime `USE_TF32` flag (default True, disable with `-D LLMM_NO_TF32=1`).
For `dtype == float32 and USE_TF32`, use `COMPUTE_32F_FAST_TF32`; otherwise `COMPUTE_32F`. Same
pattern applied to `llmm/attention.mojo`'s batched GEMM.

**Code sites:**
- `llmm/matmul.mojo:_matmul_cublaslt` — compute type selection (goal1/fp32-parity branch;
  may not be merged to goal2/fp8-training yet).
- `llmm/attention.mojo:_attn_gemm_batched` — same pattern.

**Test gating:** Goal1 introduced `make verify-gpu-tf32` (TF32 ON) with a calibrated tolerance
(0.02, ~2× measured max drift of 0.0102) and `make verify-gpu` (TF32 OFF via `-D LLMM_NO_TF32=1`,
true IEEE fp32). Both gates run; the TF32 gate accepts larger loss/gradient differences because
TF32 arithmetic is inherently less precise than IEEE. A flat blanket tolerance would have buried
three real backward bugs (see the training-correctness campaign in metal_port_gotchas.md).

**Status in goal2/fp8-training branch:** Not yet merged. This branch still has hardcoded
`COMPUTE_32F`. The FP8 campaign should not re-merge the TF32 fix opportunistically; goal1 is
on the merge queue independently.

**Source:** Goal1/fp32-parity branch commit c27a1f9 and descendants; cited in goal1 merge
message at commit 9382ee1.

### TF2 — Loss-gating pattern for numerics-changing optimizations: strict gate OFF, real gate calibrated

**Pattern:** For any optimization that changes numerics (tensor cores, TF32, later FP8 scaling
tuning), use **two** gates:
1. **Strict gate (TF32/optimization OFF):** Verify against llm.c behavior or prior-known-good
   baseline with atol/rtol calibrated to catch real errors. Goal1 uses `-D LLMM_NO_TF32=1`.
2. **Real gate (TF32/optimization ON):** Run with the optimization enabled, using a tolerance
   calibrated to ~2× measured drift in controlled conditions (not a cranked blanket tolerance).

Goal1 measured TF32 impact in isolation: max gradient drift = 0.0102, loss drift = 0.0121.
The real gate uses atol=0.02 (≈2×), and a strict gate using atol=0.002 (≈0.2×) catches bugs
the real gate misses.

**Anti-pattern:** A flat `atol=2.0` (the old `test_gpt2` default) is a footgun. It passes
correct values and incorrect ones alike. Goal1's test hardening (metal_port_gotchas.md C1–C3)
found three backward bugs the old tolerance completely missed.

**Source:** Goal1/fp32-parity, commit 8604a9d and descendants.

### TF3 — Benchmark hygiene on GB10 (unified memory): interleave A/B arms in one quiet window

**Gotcha:** GB10's unified memory and the concurrent vLLM tenant can produce phantom regressions.
A stacked train + vLLM run produced a false 732 ms CPU regression (2026-07-03 before lockfile
discipline). The fix: **flock discipline** (`flock /tmp/llmm-gpu.lock` wrapping all GPU commands)
and **interleaving A/B comparison arms in one quiet window**.

**Hygiene checklist:**
- GPU lockfile: every GPU-touching command uses `flock -w 10800 /tmp/llmm-gpu.lock -c '...'`.
- Baseline check: `free -g` and `nvidia-smi` before any timing run; confirm idle GB memory.
- Interleave: A/B arms run back-to-back in one locked session, no restarts between.
- Sanity check: `tok/s = B*T/ms` in every timing table (catches obvious measurement bugs).
- Thermal: GB10 throttles less severely than M4 Max (no jetsam), but allow ~30 sec idle between
  unrelated measurement sessions.

**Source:** Memory footprint contention notes (MEMORY.md, worktree-agents-must-commit context).

### TF4 — Wall-clock A/B benchmarks cannot resolve sub-millisecond thermal-drift effects; alternate arm order across rounds

**Gotcha:** Naive fixed-order A/B benchmarking (always run A then B in every round) produces
false **apparent wins/losses** of ±3–4 ms when the true effect is <0.5 ms, because thermal state
drifts predictably over minutes — an A-first warm B-second-cold pair will always make B look
slower even if they are identical. A 60-sample interleaved benchmark (3 rounds × 20 steps/round)
with fixed order showed +4.17 ms false win; a second run alternating which arm went first
(A-first then B-first each round) showed −1.10 ms / +0.17 ms — opposite sign and noise-amplitude.

**Fix:** Alternate arm order across rounds (e.g., round 1: A then B, round 2: B then A, round 3:
A then B). Per-arm standard error is ~0.2–0.3 ms, so effects smaller than that cannot be
resolved reliably; use per-kernel ncu profiles (`make profile-ncu PROFILE_T=1024`) for
sub-millisecond targeting instead.

**Source:** 2026-07-10 dbias-fusion wall-clock interleaving study
(`docs/ai/ai_assisted_optimizations_and_benchmarks.md`).

### TF5 — `make benchmark-gpu` silently drops PyTorch arms if the default pixi env has no CUDA torch

**Gotcha:** `scripts/benchmark_train.py` in the default pixi environment fails PyTorch CUDA arms
with "Torch not compiled with CUDA enabled", then silently **drops those samples from the output
table** rather than erroring — a 6-arm table becomes a 4-arm table invisibly. The fix: invoke
`pixi run -e cuda python scripts/benchmark_train.py ...` to use the CUDA-enabled torch environment.

**Source:** 2026-07-10 GPU 6-arm official benchmark (`docs/ai/ai_assisted_optimizations_and_benchmarks.md`,
Harness gotcha section).

---

## ARCHITECTURE DECISIONS

### A1 — FP8 is transient GEMM-operand-only; storage stays bf16 + fp32 master

**Decision:** Under FP8 (and FP4), **model storage is unchanged** — parameters, activations,
gradients stay bf16 (`GPT2_DTYPE = bfloat16`); optimizer master weights stay fp32.
**fp8 exists only as short-lived device quantized copies of the two operands feeding a linear
GEMM**, plus their scale/amax scalars. The GEMM consumes fp8 and writes bf16 output.

**Why this layout (and why it is also the right FP4 layout):**
- **Neutralizes landmine #2 (host↔device element-size mismatch).** No host buffer changes element
  size. The fp8/fp4 buffers are device-only transients never read back to a typed host buffer
  in the normal path.
- **Keeps LayerNorm, softmax, attention core (QKᵀ / softmax·V), GELU, residual adds, embeddings,
  the LM head, cross-entropy, and AdamW bit-for-bit identical to bf16** — matching Transformer-Engine's
  HYBRID recipe, which keeps exactly these ops in high precision.
- **No forked forward/backward.** `GPT2.forward` / `GPT2.backward` / `GPT2.update` are unchanged.
- **FP4 reuses it verbatim** — storage still bf16, only the transient operand dtype + scaling
  granularity change.

**Source:** `docs/ai/fp8_training_design.md` (this branch), §1.1.

### A2 — One LLMM_PRECISION axis (fp32|bf16|fp8|fp4) with LLMM_BF16 back-compat alias

**Decision:** Introduce a single ordered axis `LLMM_PRECISION` in `train_gpt2.mojo` replacing
the `USE_BF16` / `GPT2_DTYPE` block. Values: `"fp32"`, `"bf16"`, `"fp8"`, `"fp4"` (future).
`-D LLMM_BF16=1` is a back-compat alias for `LLMM_PRECISION=bf16`.

**Rationale:** fp8 and fp4 are mutually exclusive; a growing set of independent booleans is
error-prone and violates DRY. One axis is extensible.

**Comptime structure:**
```mojo
comptime PRECISION = _resolve_precision()  # reads LLMM_PRECISION / LLMM_BF16
comptime LOWP_ENABLED = PRECISION == "fp8" or PRECISION == "fp4"
comptime STORAGE_DTYPE = DType.float32 if PRECISION == "fp32" else DType.bfloat16
comptime GPT2_DTYPE = STORAGE_DTYPE
comptime USE_BF16 = STORAGE_DTYPE == DType.bfloat16   # master iff bf16
```

**Implication:** `USE_BF16` keeps its exact current meaning, so `llmm/adamw.mojo`, `zero.mojo`
need **no change**. fp8/fp4 get the fp32 master path for free.

**Gotcha:** Inconsistent combinations (e.g., both `LLMM_BF16=1` and `LLMM_PRECISION=fp8`) and
unknown precision values are **comptime errors**, not silent fallbacks.

**Source:** `docs/ai/fp8_training_design.md` (this branch), §2; reported by campaign agent
(chunk A, commit 804b10d), not yet fully deployed/tested on all targets. Mark as "agent-reported,
awaiting Chunk B integration."

### A3 — Worktree/agent process conventions for safe parallel campaigns

**Convention:** One worktree per agent (shared tree = commit races), sub-branch per implementation
chunk with disjoint file ownership, coordinator does all merges, GPU flock for every GPU-touching
command.

**Example:** goal2/fp8-training has sub-branches for each chunk (goal2/fp8-chunk-a, goal2/fp8-chunkB,
etc.), each pair-independent so multiple agents can work in parallel without merge conflicts.
Coordinator merges to goal2/fp8-training when chunks complete.

**Source:** Worktree-agents-must-commit (MEMORY.md context); goal2 branch structure.

---

## FP8 CHUNK D/E (forward + backward integration)

### E1 — fp8/bf16 builds MUST load the bf16 checkpoint, not the fp32 one

**What:** Storage stays bf16 under `LLMM_PRECISION=fp8` (A1), so any fp8 or bf16 build reads
`gpt2_124M_bf16.bin` (EXPECTED_VERSION=5), never `gpt2_124M.bin` (fp32). Tools select via
`comptime checkpoint = "gpt2_124M_bf16.bin" if GPT2_DTYPE == DType.bfloat16 else "gpt2_124M.bin"`.
Loading the fp32 checkpoint into a bf16-storage model mismatches element sizes on readback
(cf. MEMORY.md `bf16-loss-buffer-dtype-mismatch`). The `dump_grads_gpt2.mojo` gate tool and any
new fp8 harness must follow this.

### E2 — `List[AmaxState]` requires an explicit `(Movable)` declaration

**What:** `train_gpt2.mojo`'s `LowpState` holds a `List[AmaxState[FP8_SPEC]]` per site per layer.
Mojo does not infer `Movable` conformance even when every field is movable, so `AmaxState` must
declare `struct AmaxState[spec](Movable)`. Symptom without it: `List[AmaxState]()` fails with
"has 'Movable' type, but value has type 'AnyStruct[AmaxState]'". No `__moveinit__` needed — the
compiler-synthesized move is correct.

### E3 — Sibling-function convention for the fp8 GEMM path (Chunk D/E)

**What:** fp8 GEMMs are separate entry points (`matmul_fwd_lowp`, `matmul_bwd_lowp`,
`matmul_d_input_bwd_lowp`, `matmul_d_weight_bwd_lowp`) called from a `comptime if LOWP_ENABLED:`
branch, with the unmodified bf16 `matmul_fwd`/`matmul_bwd` in the `else:`. Rationale: the bf16
signatures have no room for the per-site `AmaxState`s, and under bf16/fp32 the `else:` branch is
the ONLY branch elaborated (byte-identical to pre-chunk code — the comptime-gating proof, gate b).
Do not add fp8 parameters to the shared functions.

### E4 — AmaxState single-update ownership (the "once per step" contract)

**What:** Each `AmaxState.update_scale` call pushes this step's amax into the ring buffer;
calling it twice for the same operand in one step double-pushes and poisons the delayed-scaling
history. Ownership: **forward** (`matmul_fwd_lowp`) updates the input/weight (E4M3) states exactly
once; **backward** reads those same states read-only (dgrad/wgrad only re-quantize the same bf16
tensors at the already-current scale). The backward-only `d_output` (E5M2) state has no forward
counterpart, so `matmul_bwd_lowp` updates it exactly once, before either sub-GEMM, and both dgrad
and wgrad read the result. Warmup (`step < amax_history_len=16`) uses the current step's own amax;
a never-updated state falls back to `scale=1.0` (only legitimately hit by an all-zero/NaN tensor).

### E5 — INVESTIGATION (2026-07-10): the failing per-tensor gradient gate is EXPECTED error compounding, not a bug

**Question:** Chunk E's per-tensor gradient gate (`dump_grads_gpt2.mojo` + `tests/compare_grad_dumps.py`,
cos>0.99 && relL2<0.1 over all 148 param-grad tensors, fp8 fwd+bwd vs bf16 reference, one step on
`gpt2_124M_debug_state.bin`) passes only 9/148 (median cosine 0.987, median relL2 0.174). Real bug or
expected quantization compounding?

**Verdict: COMPOUNDING (expected).** Four independent lines of evidence:

1. **State-ordering audit (static): clean.** Forward updates input/weight states once before backward
   reads them; `d_output` state updated exactly once in `matmul_bwd_lowp`; no double-update; margin=0,
   16-step history; at step 0 every site is in warmup so it quantizes at its OWN current-tensor amax
   (best case, no stale scale, no `scale=1.0` fallback). A state bug would have surfaced as one
   catastrophically-wrong site class — none observed.

2. **Forward/backward error split** (added a temporary `-D LLMM_FP8_FWD_ONLY=1` comptime knob that
   forces the bf16 backward else-branch): forward-fp8-only vs bf16 already gives **median relL2 0.156**
   (15/148 pass) — ~90% of the total error. Backward's incremental contribution (full vs fwd-only) is
   **median relL2 0.061** (125/148 pass on its own). The two combine in quadrature:
   `sqrt(0.156² + 0.061²) = 0.167 ≈` observed 0.174. Chunk E's backward code is NOT the culprit; the
   error is dominated by the forward fp8 GEMMs that Chunk D already shipped and gated.

3. **Depth scaling: the backprop-compounding fingerprint.** All 12 per-layer tensor classes show a
   strong negative correlation between layer index and relL2 (corr −0.68 to −0.97): error is worst at
   EARLY layers (near input) and smoothly decreases toward the output. Gradients accrue quantization
   error multiplicatively as they flow output→input — the exact expected signature. A bug would spike
   one layer/site; instead the curve is monotonic and coherent across every class. (Worst tensors:
   `ln_1_gamma`, `wpe` — small, spiky, reduction-heavy grads sensitive to relative error; note
   `ln_1_gamma` grads come from the bf16 LN-backward kernel, so their error is purely propagated
   upstream fp8 contamination, not a kernel defect.)

4. **e5m2/e4m3 per-tensor quantization is well-behaved at step 0** (numpy characterization on the
   reference grads): per-tensor e4m3 relL2 median 0.026, e5m2 median 0.052 (2 vs 3 mantissa bits, as
   expected); saturation fraction ~0 (max 0.13%) — per-tensor amax scaling maps the largest element to
   exactly fmt_max so nothing overflows; not outlier-dominated in any harmful way (even `wte` with
   amax/p99.9≈5744 quantizes to relL2 0.044). The per-GEMM floors (~0.026 E4M3 / ~0.052 E5M2) compound
   across ~12 layers to the observed ~0.17.

**Functional gate (d): PASS.** fp8 vs bf16 10-step training (fineweb shard, b=4 t=1024, identical
invocation) tracks within noise — no NaN/Inf (the "(nanz)" token is a warmup z-score, not a NaN):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | val |
|------|---|---|---|---|---|---|---|---|---|----|-----|
| fp8  | 3.755 | 4.339 | 4.236 | 3.578 | 3.574 | 3.591 | 3.220 | 3.683 | 3.563 | 3.436 | 3.534 |
| bf16 | 3.739 | 4.230 | 4.223 | 3.545 | 3.569 | 3.594 | 3.204 | 3.654 | 3.545 | 3.423 | 3.525 |

**Recalibration proposal (for the coordinator — NOT unilaterally applied to the design gate):**
The flat per-tensor `relL2<0.1 && cos>0.99` demands per-tensor error below the single-GEMM floor while
gradients traverse up to ~12 fp8 GEMMs — it is structurally unsatisfiable for a real fp8 training step
and buries no signal (cf. MEMORY.md `weak-gates-overrule-nothing`: a gate must be able to catch the
failure it guards). Propose replacing it with a depth-aware envelope + a functional primary:
  - **(a) cosine floor per tensor** (≈ >0.93): catches gradient-direction reversal, the true
    optimizer-health failure a real bug would cause.
  - **(b) relL2 envelope** (not per-tensor pass/fail): median <0.20, max <0.50, calibrated to the
    per-GEMM-verified floors (0.036 E4M3 / 0.052 E5M2) × depth.
  - **(c) depth-monotonicity check**: flag any single layer whose relL2 exceeds ~2× the local depth
    trend — this is what actually catches a per-site quantization/state bug.
  - **(d) the 10-step fp8-tracks-bf16 loss gate as the primary acceptance criterion** (functional,
    robust, already passing).

**Tooling added:** `tests/compare_grad_dumps.py` (the comparison script referenced by
`dump_grads_gpt2.mojo`, cosine + relL2 per tensor), and the `-D LLMM_FP8_FWD_ONLY=1` fwd/bwd isolation
knob in `train_gpt2.mojo` (default off = full fp8, comptime-inert).

**Source:** Chunk E gradient-gate root-cause investigation, 2026-07-10 (worktree goal2/fp8-chunkE).

---

## FP8 CHUNK F (end-to-end verification, determinism, perf)

### FF1 — The recalibrated gradient gate is now codified; local-trend window choice matters

**What:** `tests/compare_grad_dumps.py` implements the E5 recalibration proposal as an
executable gate (`make verify-fp8-grads`): (a) per-tensor cosine floor >0.93, (b) relL2
envelope (median<0.20, max<0.50), (c) depth-monotonicity (flag any layer whose relL2
exceeds ~2× its class's local-neighbor trend), (d) NaN/Inf sentinel. All four PASS on
the current `goal2/fp8-training` HEAD (cosine min 0.9366/median 0.9870/max 0.9993;
relL2 min 0.0412/median 0.1742/max 0.4616; 0 depth violations; 0 nonfinite).

**Gotcha:** criterion (c)'s result is sensitive to the neighbor-window radius. A
radius=2 window (median of up to 4 neighbors per point) flags one borderline case,
`ln_1_gamma_layer04` at 2.04× (just over the ~2× threshold) — this is the exact
tensor E5 already characterized as smallest-magnitude/most error-sensitive
(propagated bf16-LN-backward error, not a kernel defect), and the wider window
dilutes the "local" comparison with layers further from the test point on a curve
that is monotonic-but-noisy (not perfectly smooth; E5's own depth-correlation was
−0.68 to −0.97, not −1.0). The shipped implementation uses radius=1 (immediate
neighbors only — the tightest, most standard reading of "local," and a real
signal-detection choice, not a threshold hand-tuned to pass): with radius=1 the same
point is 1.90×, under threshold. Future agents changing this gate should keep radius
small (1) rather than widen it — a wider window trades bug sensitivity for false
positives on the naturally-noisy compounding curve.

### FF2 — Bit-stability confirmed: fp8's new kernels are deterministic

**What:** Two identical fp8 10-step runs (same seed/checkpoint/data invocation)
produce bitwise-identical `step N | loss ... | norm ...` sequences (`diff` empty).
This confirms the design's expectation (§1.3): all scale/amax math is delayed
(computed from *prior* steps' history, not a same-step reduction racing the GEMM),
and the amax/quantize kernels (`llmm/amax.mojo`, `llmm/lowp.mojo`) use no shared
mutable state across threads that would introduce atomics-order nondeterminism — in
contrast to the known bf16 atomics-in-non-lowp-kernels wiggle source (MEMORY.md
territory), which fp8 does not add to.

### FF3 — 50-step envelope: no drift as amax history fills

**What:** A 10-step horizon (E5) cannot distinguish "tracks bf16" from "slowly
drifts as the 16-step amax history transitions out of warmup." A 50-step fp8-vs-bf16
run (same checkpoint/data/B=4/T=1024) shows per-step `|relative loss delta|` median
0.57%, max 1.81% (step 46), with first-half (steps 2–25) median 0.55% vs second-half
(steps 26–50) median 0.58% — flat, no growing-envelope trend. Zero NaN/Inf either arm. `AmaxState` is runtime-only (not part of the checkpoint),
so this run's amax history starts empty at step 1 regardless of the model weights
being pre-trained — the warmup-to-delayed-scaling transition (`amax_history_len=16`,
§1.3) falls around step 16-17 of this 50-step window, and the flat first-half vs
second-half median confirms it does not destabilize the loss trajectory.

### FF4 — FP8 is currently slower than bf16 at real scale; overhead is almost entirely `quantize_transpose`

**What:** At B=4, T=1024 (the shipped training config), fp8 is **~36% slower** than
bf16 (183.7 vs 134.7 ms/step median, two independent measurement protocols agreeing
to within 0.5%) — larger than the ~5-6% toy-scale estimate from an earlier probe.
A 1-step `ncu` breakdown at the matching T=1024 (the profiler's `PROFILE_T` defaults
to 64, which is **not** representative — rerun with `PROFILE_T=1024` for any future
fp8 perf work) shows why: the actual fp8 tensor-core GEMMs are faster than bf16's
(−13.7% GEMM compute time, confirming the hardware works as expected), but a new
`quantize/amax/scale` kernel family costs 32.8% of total fp8 GPU time — more than
5× the GEMM saving. **`quantize_transpose` alone is 24.2% of total fp8 time**,
larger than any single existing kernel including `adamw_update`. The
amax-reduction/scale-update kernels (`amax_partial`/`amax_aggregate`/
`update_scale`/`amax_state_init`) are cheap by comparison (3.3% combined) and are
not the lever. See `docs/ai/ai_assisted_optimizations_and_benchmarks.md`
(2026-07-10 FP8 entry) for the full per-kernel table and the future-optimization-pass
roadmap (fuse the transpose into quantize's write pattern, or eliminate the operand
orientation that needs a transpose at all).

**Source:** Chunk F end-to-end verification, 2026-07-10 (worktree `goal2/fp8-training`,
`llmm-goal2-fp8`; `docs/ai/fp8_training_design.md` §6 Chunk F gates).

---

## FP8 quantize-family optimization (goal2/fp8-quant-opt branch)

### G1 — no passwordless sudo on this box means no ncu hardware counters; use timing-only comparisons instead

**What:** `ncu --metrics dram__bytes_read.sum,...` (and `--sudo`) require elevated
GPU performance-counter access; this box's driver gates that to admin users and
there is no passwordless `sudo` available (`sudo -n true` fails). Without it, ncu's
DRAM/tensor/occupancy/stall columns are blank ("n/a") — only per-kernel wall time
and call counts are available (same limitation Chunk F's own profile silently hit).

**Workaround:** for a memory-access-pattern diagnosis (e.g. "is this kernel's store
uncoalesced?"), compare wall-clock time between two kernels that process the *same*
tensor identities at the *same* element count with *identical* per-element compute,
differing only in the access pattern under suspicion. `quantize_transpose` vs.
`quantize` (same weight/input/d_output tensors, same `encode_fp8` per-element math,
differing only in whether the store is transposed) gave a clean 4.3-4.9x gap
attributable entirely to the write pattern — as diagnostic as an occupancy/stall
table would have been for this specific question, just derived differently. Static
code reading (the write address formula) is the other free source of truth this
worktree relied on to reach a confident diagnosis without hardware counters — see
the 2026-07-10 quant-opt entry in `ai_assisted_optimizations_and_benchmarks.md`.

**Source:** FP8 quant-opt Optimization A diagnosis, 2026-07-10 (worktree
`llmm-goal2-qopt`, branch `goal2/fp8-quant-opt`).

### G2 — sharing a `DeviceBuffer` across nested function calls needs an explicit keep-alive; a single-step gate won't catch the race, only a multi-step wall-clock twin-run will

**What:** Optimization B (fusing `d_output`'s natural+transposed fp8
quantize into one dual-output kernel) created the two output `DeviceBuffer`s
in `matmul_bwd_lowp` and handed raw pointers 3 call-levels deep
(`matmul_bwd_lowp` -> `matmul_d_input_bwd_lowp`/`matmul_d_weight_bwd_lowp`
-> `lowp_gemm_devscale` -> `_matmul_cublaslt_fp8`). This is a DIFFERENT
shape than every existing scratch-buffer pattern in `llmm/matmul.mojo`,
which all create-and-consume a `DeviceBuffer` within a SINGLE function.
`DeviceBuffer.__del__`'s docstring claims release is stream-ordered ("the
actual deallocation may occur asynchronously after all operations using
this buffer have completed"), and Mojo's ASAP/destroy-at-last-use means a
buffer whose only remaining reference is inside the FIRST of two sibling
nested calls gets destroyed before the SECOND call — which itself
allocates its own scratch buffers — even runs. In practice this produced
a real, reproducible race: a first implementation passed every unit test
and the single-step `verify-fp8-grads` gate (which uses a small T=64
fixed-batch single fwd+bwd step) but a full-scale (T=1024, 12 layers,
B=4) 10-step wall-clock twin-run was NOT bit-identical — two back-to-back
runs of the identical binary diverged starting at step 4, and a third run
diverged differently again, i.e. run-to-run NONDETERMINISTIC (not merely
"a new but consistent numerics path", which would rule out a race and
just mean a legitimate behavior change). A single-step gradient dump run
twice back-to-back stayed bit-identical throughout — the race needed the
full multi-step, full-scale (larger buffers, more allocator churn from
concurrent per-layer/per-step scratch creation) training loop to surface,
not just more launches of the same shapes.

**Fix:** two trivial keep-alive reads (`_ = buf.unsafe_ptr()`) at the end
of the OUTER function, after both nested consumers have been called, so
Mojo's liveness analysis keeps the buffer referenced through the whole
function body rather than trusting the (real, but apparently
insufficient at this call depth) stream-ordering guarantee alone. Three
consecutive 10-step twin runs were bit-identical after the fix, and
matched the pre-optimization baseline exactly.

**Lesson for future cross-call buffer sharing in this codebase** (relevant
to the deferred Optimization D — fusing weight/input's redundant pair
across the forward/backward call boundary, a much deeper and longer-lived
version of the same pattern): (1) any `DeviceBuffer` created in one
function and consumed by a DIFFERENT (nested or sibling) function is a
signal to add an explicit keep-alive at the creating function's end,
covering every consumer, rather than relying on stream-ordered `__del__`
alone; (2) a single-step, fixed-batch gate (`verify-fp8-grads`) is
NECESSARY but not SUFFICIENT for this class of bug — always additionally
run a multi-step, full-scale (matching the real training config, not the
profiler's small default shapes) wall-clock twin-run before considering
a memory-layout/lifetime-touching optimization done; (3) if a twin-run
ever diverges, run a THIRD time before concluding anything — two runs
alone cannot distinguish "consistently different because of a real (buggy
or intentional) code-path change" from "randomly different because of a
race", but three runs where none agree pairwise consistently (or where
repeated re-runs keep landing on different trajectories) is strong
evidence of the latter.

**Source:** FP8 quant-opt Optimization B, 2026-07-10 (worktree
`llmm-goal2-qopt`, branch `goal2/fp8-quant-opt`).

## FP4 CHUNK T1 (forward-pass integration)

### F1 — Cosine and relL2 gate thresholds are geometrically coupled: never mix floors across precisions

**What:** Chunk T1's gate-(c) test (`tests/test_matmul_fwd_fp4.mojo`) initially failed both MLP
sites (fc cosine 0.99030, proj 0.98944 vs a `>0.999` floor) while comfortably passing its relL2
bound (0.151 < 0.20). The cosine floor had been inherited from the fp8 sibling test
(`test_matmul_fwd_lowp.mojo`), where it is self-consistent with fp8's ~0.036 relL2 floor. It is
**mathematically unreachable** at fp4's calibrated ~0.151 relL2 floor: for quantization noise
roughly orthogonal to the signal, `cosine ≈ 1/sqrt(1 + relL2²)`, so relL2 0.151 caps cosine at
~0.9886 — matching the measured 0.989–0.990 almost exactly (bit-identical across three runs).

**Rule:** when writing a two-metric (cosine + relL2) accuracy gate, derive the cosine floor FROM
the calibrated relL2 floor via `1/sqrt(1+relL2²)` minus margin — a copy-pasted cosine floor from
a higher-precision sibling silently demands sub-floor error. Conversely, if measured cosine lands
much LOWER than the geometry predicts from measured relL2, the error is correlated with the
signal (systematic bias, e.g. a scale bug), not quantization noise — a real red flag even when
both raw numbers look "close to 1". Recalibrated to 0.985 in commit `cac1968`
(coordinator-approved, per the campaign's calibrate-gates-against-measurement precedent).

**Gate (d) note (fp4 fwd-only cost):** 10-step fp4-fwd/bf16-bwd training vs bf16 shows mean
per-step loss delta +0.83%, final val +1.26% (4.547→3.756 vs 4.513→3.709, both decreasing, no
NaN/Inf) — a real but small degradation, unlike fp8 which tracked within noise. This is the
expected e2m1 RNE forward cost; chunk T2 (SR + RHT + fp4 backward) owns the full convergence
envelope. Full tables in `ai_assisted_optimizations_and_benchmarks.md` (2026-07-10 entry).

**Source:** Chunk T1 gate validation, 2026-07-10 (worktree goal3b/fp4-training, commits
135bf27 + cac1968).

---

## FP4 CHUNK T2a (backward pass — SR only, no RHT yet)

### F2 — Comparing two independently-allocated swizzled scale buffers byte-for-byte requires zeroing padding first

**What:** A dedicated correctness test for the new `nvfp4_quantize_transpose`
(compare it against materializing the transpose in bf16 host-side and calling
plain `nvfp4_quantize` on the result, expecting byte-identical output) FAILED
on its first run: the packed e2m1 data buffer (768/768 bytes) matched
perfectly, but the swizzled e4m3 scale buffer diverged at byte offset 2
(201 vs 123) despite both calls using identical `rows`/`k`/`BLOCK_ROWS` and
therefore identical swizzle geometry.

**Root cause:** the swizzled scale buffer's cuBLAS 128-row/4-col tile layout
has PADDING — for a `[48, 32]` logical scale tensor (96 real 1x16-block
scale values), `nvfp4_swizzled_scale_buffer_size` allocates 512 bytes, of
which only 96 are ever written by the quantize kernel (byte offset 2 solves
to no valid `(row, col)` pair under the swizzle formula — it falls in an
unused padding slot). `ctx.enqueue_create_buffer` does not zero-initialize,
so two SEPARATELY allocated scale buffers have independently garbage
padding bytes that predictably fail a byte-for-byte comparison — a test-
harness bug, not a kernel bug (the actual quantize logic, including the new
`TRANSPOSE`-flag source-read address swap, was correct all along).

**Fix:** `ctx.enqueue_memset(scale_buf, Scalar[DType.uint8](0))` on BOTH
scale buffers before running either quantize call, so never-written padding
compares `0 == 0` instead of garbage vs garbage. After the fix, all 768
packed bytes AND all 512 scale bytes (padding included) matched exactly,
confirming `nvfp4_quantize_transpose` reproduces `nvfp4_quantize`-on-the-
materialized-transpose bit-for-bit, as the shared-kernel-body design
predicts.

**Lesson (generalizes C5's swizzle-layout lesson):** any test that compares
two independently-allocated buffers byte-for-byte, where the buffer's
logical content doesn't fill 100% of its physical layout (swizzle/tile
padding, alignment padding, etc.), must zero (or otherwise pin) the padding
region first — a byte-identity assertion is only meaningful over the bytes
BOTH sides' producers actually write.

**Source:** `tests/test_nvfp4_quant.mojo::
test_quantize_transpose_matches_materialized_transpose_gpu`, Chunk T2a,
2026-07-10 (worktree goal3b/fp4-training).

### F3 — Dgrad/Wgrad's fp4 accuracy floor is worse than Fprop's single-GEMM floor, and that's expected (not a new instance of E5's compounding, but a related composition effect)

**What:** Chunk T2a's backward-GEMM gate (`tests/test_matmul_bwd_fp4.mojo`)
measures relL2 ≈ 0.178–0.184 / cosine ≈ 0.985 for `d_input`/`d_weight` at
both MLP sites — noticeably worse than T1's forward gate's ~0.151 relL2 /
~0.989 cosine floor at the SAME shapes and comparable data distributions.

**Why (not a bug):** each backward GEMM quantizes TWO operands through a
transposed-quantize path (weight/input RNE, `d_output` SR) — RNE's rounding
error and SR's dither variance both feed the same GEMM, whereas the forward
GEMM's ~0.151 floor is from two RNE-only quantizations. SR is unbiased
(`E[quantize(x)] == x`) but NOT lower-variance than RNE — trading bias for
variance is the whole point of stochastic rounding for gradient
accumulation, and it shows up here as a slightly higher single-GEMM relL2,
exactly as `fp4_training_recipes_research.md`'s "FP4 All the Way" citation
(the √3 gradient-noise-threshold paper) predicts.

**Gate calibration:** relL2 < 0.22 / cosine > 0.975 (derived via `cosine ≈
1/√(1+relL2²)` from the measured floor, same F1-gotcha methodology, NOT
copy-pasted from T1's forward bound).

**Source:** Chunk T2a gate (c) validation, 2026-07-10 (worktree
goal3b/fp4-training).

### F4 — T2a's 10-step training gate is a REGRESSION vs T1's fwd-only gate, as the recipe predicts pre-RHT

**What:** fp4 (fwd+bwd, SR, no RHT) vs bf16 10-step training (tinyshakespeare,
checkpoint init) shows mean per-step loss delta +1.08% (final val +1.50%),
WORSE than T1's fwd-only +0.83% (final val +1.26%). Step 1's loss is
bit-identical between the T1 and T2a runs (4.408221 both) — a strong sanity
check that both runs share the same seed/data/init and that the divergence
from step 2 onward is genuinely attributable to the backward-pass change,
not a harness difference.

**Why this is expected, not a regression to fix:** `fp4_training_recipes_
research.md`'s mandatory recipe has THREE ingredients (SR on gradients, RHT
on Wgrad, 2D/1D block scaling) and this chunk (T2a) ships only the first.
NVIDIA's own ablations report RHT is essential for Wgrad accuracy — without
it, outlier-heavy activation/gradient channels absorb more quantization
error. The coordinator's task brief explicitly predicted this ("expect
worse before RHT — report honestly"). T2b (RHT on Wgrad, per the recipe) is
expected to close most of this gap.

**STOP condition check:** NOT triggered — no NaN/Inf, loss decreases
monotonically-ish in both arms (same noisy-but-decreasing shape as T1's
run), and no individual gate failed by >2x its calibrated bound (T2a's own
+1.08%/+1.50% are the MEASUREMENT, not a bound being exceeded — there is no
pre-declared numeric ceiling for gate (d), only "report honestly").

**Source:** Chunk T2a gate (d), 2026-07-10 (worktree goal3b/fp4-training);
full table in `ai_assisted_optimizations_and_benchmarks.md`'s 2026-07-10 T2a
entry.

---

## FP4 CHUNK T2b (Wgrad Random Hadamard Transform)

### F5 — A single dominant outlier in an RHT block gets SPREAD, not shrunk — e2m1's coarse ladder can lose more from that than it gains

**What:** A dedicated unit test (`tests/test_matmul_bwd_fp4.mojo::
test_wgrad_rht_outlier`) constructs a 16-block containing 15 ordinary
(`~N(0,1)`) values and one isolated 100x-scaled "outlier" value (0.1% of all
entries, matching the coordinator's literal spec), then compares
`matmul_d_weight_bwd_fp4` RHT-on vs a hand-composed RHT-off arm against a
bf16 reference. Naive expectation (and the module docstrings' framing,
"Gaussianizes outliers so a block quantizes with less error"): RHT-on should
be strictly better here — this is supposedly RHT's motivating case.
Measured (bit-identical across repeat runs): RHT-on is actually WORSE
(rel_l2 0.0969 vs RHT-off's 0.0844).

**Why (not a bug):** the unnormalized 16x16 Hadamard has every entry `+/-1`,
so mixing one dominant value `V` into an otherwise-small block produces
`y_i = +/-V + O(sqrt(15))` for EVERY output position `i` — the block's PEAK
magnitude does not shrink (still `~V`), it gets SPREAD from 1-of-16 slots to
all 16. Pre-RHT, e2m1's 8-level ladder crushes the 15 small values to
exactly 0 (a real information loss) but encodes the 1 large value with good
relative precision. Post-RHT, all 16 values are comparable in magnitude to
`V` but only differ from each other by the `O(sqrt(15))` mixing noise — a
small RELATIVE spread that e2m1's coarse ladder (top two nonzero levels are
4 and 6, a 50% gap) cannot resolve, so most of the 16 transformed values
collapse to the SAME quantized code, losing the very differentiation the
GEMM needs to reconstruct each element's true contribution.
`docs/ai/fp4_training_recipes_research.md`'s own citation of the Metis paper
(arXiv:2509.00404) makes exactly this point: RHT "only smooths a few
outliers without reducing overall spread" — this test reproduces that
critique empirically, in-repo, at the single-GEMM level.

**Scope of the finding:** this is a property of an ISOLATED single-spike
outlier construction, not necessarily of realistic multi-outlier/correlated
gradient structure (the recipe's actual target, and what NVIDIA's
large-scale ablations were run against). A companion test on plain gaussian
data (`test_wgrad_rht_gaussian`) shows RHT-on roughly neutral-to-slightly-
better (rel_l2 0.1674 -> 0.1664), and the real MLP-shaped site gates
(`test_fc_bwd_site`/`test_proj_bwd_site`, uniform random data) show a real
~7-10% relL2 improvement (0.178-0.184 -> 0.166 for `d_weight`) — so RHT is
not uniformly harmful, just not uniformly helpful either. See F6 below for
what this means for the end-to-end training gate.

**Source:** Chunk T2b gate (c)/ablation unit tests, 2026-07-10 (worktree
goal3b/fp4-training).

### F6 — T2b's RHT integration is verifiably correct but provides negligible end-to-end benefit at this scale/setup — report honestly, do not force the recipe's narrative

**What:** Per the recipe (`fp4_training_recipes_research.md` §1), RHT on
Wgrad is supposed to "close most of the gap" T2a's SR-only regression opened
(+1.08%/step, +1.50% final val vs bf16). Measured 10-step training
(tinyshakespeare, B4 T1024, GPT-2 124M checkpoint init, same invocation as
T1/T2a): **T2b (RHT-on) mean per-step delta +1.06%, final val +1.50%** —
essentially IDENTICAL to T2a's +1.08%/+1.50%, not a meaningful recovery.

**This is not a wiring bug — three independent checks confirm the RHT path
is doing exactly what it's supposed to:**
1. The RHT-composition contract test
   (`tests/test_lowp_gemm_fp4.mojo::test_fp4_rht_quantize_gemm_contract`,
   unaffected by this chunk's changes, still PASSES) confirms
   `(H@a)^T@(H@b) == 16*a^T@b` holds through quantize+GEMM.
2. `-D LLMM_FP4_NO_RHT=1` (the ablation build) reproduces T2a's ORIGINAL
   10-step per-step losses BIT-FOR-BIT (all 10 steps + val loss match to
   every printed digit) — proof the ablation flag correctly falls back to
   T2a's exact code path with nothing else disturbed.
3. The dedicated GEMM-level unit tests (F5 above) show RHT DOES measurably
   improve `d_weight` accuracy on realistic MLP-shaped data (~7-10% relL2
   reduction) — the mechanism works, it's just a small effect at the
   single-GEMM level that doesn't compound into a visible training-loss
   difference over only 10 steps.

**Why the recipe's claim doesn't clearly materialize here (informed
speculation, not verified against NVIDIA's own ablation setup):** NVIDIA's
"RHT closes most of the gap" claim comes from LARGE-SCALE (billions of
tokens, larger models) from-scratch pretraining ablations, where (a) small
per-step accuracy deltas compound over far more steps than this gate's 10,
and (b) gradient outlier/heavy-tail structure is a from-scratch-training
phenomenon that may not be well-represented by only 10 steps starting from
an already-converged checkpoint (this gate's `-e gpt2_124M_bf16.bin` init,
chosen for determinism/speed, not gradient realism) — see F5's finding that
RHT's benefit is genuinely data-distribution-dependent (helps on realistic
random data, roughly neutral on gaussian, can even hurt on an isolated
single-spike outlier).

**STOP condition check:** NOT triggered — no NaN/Inf in either arm, loss
decreases in both, ablation reproduces T2a bit-for-bit (proving nothing else
broke), and no gate failed by >2x its calibrated bound. This is a "measure
honestly, don't force the hoped-for narrative" finding, same category as
F3/F4/E5 — the coordinator's brief explicitly anticipated this possibility
("report the measured mean delta honestly, whatever it is").

**Source:** Chunk T2b gate (d) + ablation consistency check, 2026-07-10
(worktree goal3b/fp4-training); full table in
`ai_assisted_optimizations_and_benchmarks.md`'s 2026-07-10 T2b entry.

---

## FP4 CLOSEOUT (bit-stability hardening, perf snapshot, transpose-coalescing verdict)

### F7 — A freshly-rebuilt binary's FIRST invocation can diverge from its own later invocations; this is not an SR/seed bug and is not fp4-specific

**What:** The closeout's bit-stability re-check ran two back-to-back fp4
10-step twin runs immediately after a fresh `make build-fp4` (as part of
gating the F8 transpose-coalescing optimization below) and got a
DIFFERENT trajectory from run 1 to run 2 — starting at step 3 (loss
4.579138 vs 4.587136, a ~0.17% relative gap, never NaN/Inf, never
non-decreasing). This reproduced a SECOND time on a completely independent
rebuild. Both times, the divergence pattern was the SAME shape: exactly one
"odd one out" run, and it was always the FIRST invocation after
`make build-fp4` finished. 5 additional back-to-back runs against an
ALREADY-WARM binary (no intervening rebuild) were unanimous (all 5 matched
each other).

**Root cause: NOT the code being tested.** The second rebuild used
`git checkout -- llmm/matmul.mojo` to restore byte-identical, unmodified
HEAD content (confirmed via `git diff` showing zero changes) before
rebuilding — and the SAME first-invocation-diverges pattern reproduced on
this completely clean tree. This rules out any correctness bug in whatever
code happened to be under test at the time (see F8) and points instead at
something environment/build/scheduling-dependent that predates this
session's work entirely.

**Working hypothesis (disclosed as speculation, not verified against
source):** `llmm/layernorm.mojo`'s LN-backward parameter-gradient
accumulation (`_ln_dparam_accum` kernel, called every layer, every step, in
EVERY precision's build — not fp4-specific) uses a non-associative
`Atomic[DType.float32].fetch_add`. Floating-point addition is not
associative, so the exact numeric result of an atomic-accumulated sum
depends on the ORDER concurrent thread blocks perform their adds, which in
turn depends on GPU kernel-launch scheduling. A freshly built binary's
first-ever execution of certain kernel identities plausibly has enough
JIT/kernel-cache warm-up timing jitter (this box has no ahead-of-time
compiled cubins — kernels JIT on first use within a process, per the
~900s-fresh-process-JIT operating note) to occasionally perturb that
accumulation order into one of (at least) two distinct, both
individually-reproducible-once-warm, floating-point-valid trajectories.
This was not confirmed by directly instrumenting the atomic add (out of
this session's budget); it is offered as the most parsimonious explanation
consistent with every observation above (reproducible with clean code,
tied to freshness-of-build rather than run count, never NaN/Inf, both
trajectories individually stable once warm).

**Practical implication:** a bit-stability gate that does
`make build-fp4 && run && run && diff` can spuriously "fail" — not because
training is nondeterministic in the way the gate is trying to catch (an
SR/scale race), but because of this orthogonal first-touch effect. Prefer
running the bit-stability comparison against an ALREADY-EXERCISED binary
(as every prior FP4 chunk's gate — and this closeout's own PRIMARY
bit-stability result — did, which is why they all reported clean PASSes),
or explicitly discard the first post-build invocation before judging
bit-stability. Do not conflate this with G2 (fp8's buffer-lifetime race,
which needed an explicit keep-alive fix) — this failure mode reproduces on
code that has no buffer-sharing changes at all, so no code-level fix
applies; it is a property of the toolchain/scheduling environment.

**Source:** FP4 closeout, 2026-07-10 (worktree goal3b/fp4-training,
attempting the F8 optimization below; reproduced independently on
git-clean `matmul.mojo`).

### F8 — `_rht_transpose_prep`'s naive transpose has the same coalescing pathology fp8 fixed with a 32x32 tile; the fix is ready but unshipped pending sanitizer confirmation of F7

**What:** `llmm/matmul.mojo`'s `_gpu_transpose_kernel` (used by Chunk T2b's
`_rht_transpose_prep` to materialize `RHT(operand^T)` scratch for Wgrad) is
byte-for-byte the same one-thread-one-element, maximally-strided-WRITE
shape (`dst[c*rows+r] = src[idx]`, consecutive threads write addresses
`rows` elements apart) as fp8's ORIGINAL `_quantize_transpose_kernel`
before its proven Optimization-A 32×32-shared-memory-tile rewrite — and
unlike FP4's OWN `nvfp4_quantize_transpose` (which fuses a block-scoped
amax pass and 2-values/byte nibble packing into its transpose, making a
drop-in tile rewrite non-trivial), `_gpu_transpose_kernel` moves plain bf16
values with zero packing/blocking — a structurally perfect match for the
codebase's own already-proven tile kernel
(`_gpu_transpose_add_into_kernel`, P14/P15). Measured: 18,184.83 µs / 32
calls = 568.28 µs/call, the single largest fp4-specific kernel bucket in a
1-step `ncu` profile (8.5% of total GPU time alone; paired with
`hadamard16_fwd_gpu`'s 691.52 µs/call the RHT-prep pair is 18.8% of total
— MORE than the entire NVFP4 quantize/amax family, 14.4%).

**Fix implemented:** a new `_rht_transpose_tiled_kernel` (32×32 tile,
32×33-padded shared memory, coalesced read AND write, `dst[c,r]=src[r,c]`
with no cast/rounding — pure layout change) wired into `_rht_transpose_prep`
in place of `_gpu_transpose_kernel`. **Single-call correctness: perfect.**
All four fp4 test suites (`test_matmul_bwd_fp4.mojo` 5/5,
`test_lowp_gemm_fp4.mojo` 8/8, `test_matmul_fwd_fp4.mojo` 2/2,
`test_nvfp4_quant.mojo` 18/18) passed with every site-gate number (relL2,
cosine, RHT gaussian/outlier ablation) identical to pre-change values to
every printed digit.

**Multi-step gate: FAILED at first — but the failure is F7, not this fix.**
A 10-step wall-clock twin-run of the freshly-rebuilt binary diverged
starting step 3, matching F7's exact signature. Root-causing this required
reverting the change entirely (`git checkout -- llmm/matmul.mojo`) and
rebuilding from scratch: the SAME divergence pattern reproduced on the
UNMODIFIED, original naive-kernel code (see F7) — proving the tiled-kernel
change itself is not the cause of the multi-step divergence.

**Decision: reverted anyway, not shipped.** Even though the balance of
evidence points to the fix being numerically inert (F7 is orthogonal and
pre-existing), the mission's explicit gate for this class of optimization
is "bit-identical fp4 runs before/after," and that gate could not be
CLEANLY certified for this specific change without also root-causing F7
(out of budget this session). `compute-sanitizer` (confirmed present at
`/usr/local/cuda/bin/compute-sanitizer`, providing `racecheck`/`memcheck`
tools that could positively rule out a shared-memory hazard rather than
relying on circumstantial before/after comparison) was not run to
completion this session. **Re-attempt recipe for a future session:**
(1) re-apply the `_rht_transpose_tiled_kernel` diff (fully specified in the
2026-07-10 FP4-closeout doc entry and recoverable from this session's
worktree history), (2) run `compute-sanitizer --tool racecheck` over a
short (`-x 2`, single fp4-eligible layer) training run BEFORE trusting any
bit-stability comparison, (3) separately root-cause F7 first if possible
(it blocks a clean gate regardless of which fix is being evaluated), (4)
only then re-run the bit-stability/perf gates and ship.

**Lesson (extends G2):** G2 taught "a single-step gate is necessary but not
sufficient — always additionally run a multi-step, full-scale wall-clock
twin-run before considering a memory-layout-touching optimization done."
This chunk adds a corollary: when a multi-step twin-run DOES diverge on a
memory-layout change, don't assume the change is the cause — REVERT FIRST
and re-run the identical twin-run protocol on the clean baseline before
concluding anything, exactly as G2's own lesson #3 recommended ("run a
THIRD time before concluding") but extended one step further ("run the
REVERTED baseline before concluding it was your change"). Skipping that
step here would have produced a false-negative verdict on a numerically
inert optimization.

**Source:** FP4 closeout, 2026-07-10 (worktree goal3b/fp4-training,
`llmm/matmul.mojo`'s `_rht_transpose_prep`/`_gpu_transpose_kernel`; fix
implemented then reverted, git-clean at commit time).

---

## Section summary

| Section | Items | Status |
|---------|-------|--------|
| TOOLCHAIN | T1–T5 | Verified (T1–T2: Probes 1–5; T3–T5: observed) |
| cuBLASLt | C1–C5 | Verified (C1: Probe 4; C2–C3: Probe FP4; C4–C5: FP4-GEMM chunk, tests/test_lowp_gemm_fp4.mojo) |
| TF32/fp32 | TF1–TF5 | Verified (TF1–TF2: goal1 branch; TF3: GB10 incidents; TF4–TF5: benchmarking methodology from 2026-07-10 optimizations) |
| ARCHITECTURE | A1–A3 | A1–A2: design doc confirmed; A3: agent-reported |
| FP8 CHUNK D/E | E1–E5 | Verified (E5: root-cause investigation, coordinator-accepted recalibration) |
| FP8 CHUNK F | FF1–FF4 | Verified (gate codified + PASS, bit-stability PASS, 50-step envelope PASS, perf measured) |
| FP8 QUANT-OPT | G1-G2 | Verified (G1: ncu sudo-gated counters workaround; G2: race found + fixed, 3x twin-run confirmed) |
| FP4 CHUNK T1 | F1 | Verified (gate c re-run 3x bit-identical; gate d 10-step A/B) |
| FP4 CHUNK T2a | F2–F4 | Verified (F2: test fix + re-run 18/18 green; F3: gate c 3x bit-identical; F4: gate d 10-step A/B, checkpoint init) |
| FP4 CHUNK T2b | F5–F6 | Verified (F5: dedicated gaussian/outlier unit tests, reproducible; F6: gate d 10-step A/B + ablation bit-identity to T2a) |
| FP4 CLOSEOUT | F7–F8 | Verified (F7: reproduced 2x independently on git-clean code; F8: fix implemented+gated+reverted, root-caused via F7) |

---

## Unverified seed items (reported by campaign agent, not yet committed)

- **A2, comptime-error enforcement:** The `_resolve_precision` function and the exact
  comptime error behavior for inconsistent precision flags are described in fp8_training_design.md
  but not visible in the current goal2/fp8-training HEAD (they are in Chunk A, commit 804b10d,
  which may not yet be merged into this branch for full integration testing). Pending Chunk B
  completion and merge verification.

---

## Footer

Written with AI assistance (Claude Code / Haiku agent), directed by Evan Owen. Final editorial sweep
(numbering consolidation, cross-reference verification, gotchas from daily entries) on 2026-07-10.
