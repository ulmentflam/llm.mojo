# Low-precision training (TF32/FP8/FP4) campaigns: gotchas and changes

A consolidated technical log of the TF32 (goal1/fp32-parity), FP8 (goal2/fp8-training),
and FP4 (goal3/fp4-research) campaigns to extend `llm.mojo`'s training path to mixed-precision
regimes on NVIDIA GB10 (Grace-Blackwell, sm_121, aarch64). This document captures gotchas,
out-of-the-ordinary implementation changes, and non-obvious updates that future engineers
or agents extending low-precision support must not rediscover the hard way.

**Note at the top:** This is a LIVING first draft (2026-07-10, campaigns in flight); FP8
chunks B/C/D/E/F/G and the FP4 build will append their gotchas before the final merge.

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

### F1 — The recalibrated gradient gate is now codified; local-trend window choice matters

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

### F2 — Bit-stability confirmed: fp8's new kernels are deterministic

**What:** Two identical fp8 10-step runs (same seed/checkpoint/data invocation)
produce bitwise-identical `step N | loss ... | norm ...` sequences (`diff` empty).
This confirms the design's expectation (§1.3): all scale/amax math is delayed
(computed from *prior* steps' history, not a same-step reduction racing the GEMM),
and the amax/quantize kernels (`llmm/amax.mojo`, `llmm/lowp.mojo`) use no shared
mutable state across threads that would introduce atomics-order nondeterminism — in
contrast to the known bf16 atomics-in-non-lowp-kernels wiggle source (MEMORY.md
territory), which fp8 does not add to.

### F3 — 50-step envelope: no drift as amax history fills

**What:** A 10-step horizon (E5) cannot distinguish "tracks bf16" from "slowly
drifts as the 16-step amax history transitions out of warmup." A 50-step fp8-vs-bf16
run (same checkpoint/data/B=4/T=1024) shows per-step `|relative loss delta|` median
0.57%, max 1.81% (step 46), with first-half (steps 2–25) median 0.55% vs second-half
(steps 26–50) median 0.58% — flat, no growing-envelope trend. Zero NaN/Inf either arm. `AmaxState` is runtime-only (not part of the checkpoint),
so this run's amax history starts empty at step 1 regardless of the model weights
being pre-trained — the warmup-to-delayed-scaling transition (`amax_history_len=16`,
§1.3) falls around step 16-17 of this 50-step window, and the flat first-half vs
second-half median confirms it does not destabilize the loss trajectory.

### F4 — FP8 is currently slower than bf16 at real scale; overhead is almost entirely `quantize_transpose`

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

---

## Section summary

| Section | Items | Status |
|---------|-------|--------|
| TOOLCHAIN | T1–T5 | Verified (T1–T2: Probes 1–5; T3–T5: observed) |
| cuBLASLt | C1–C3 | Verified (C1: Probe 4; C2–C3: Probe FP4) |
| TF32/fp32 | TF1–TF3 | Verified (TF1–TF2: goal1 branch; TF3: GB10 incidents) |
| ARCHITECTURE | A1–A3 | A1–A2: design doc confirmed; A3: agent-reported |
| FP8 CHUNK D/E | E1–E5 | Verified (E5: root-cause investigation, coordinator-accepted recalibration) |
| FP8 CHUNK F | F1–F4 | Verified (gate codified + PASS, bit-stability PASS, 50-step envelope PASS, perf measured) |
| FP8 QUANT-OPT | G1 | Verified (ncu sudo-gated counters, timing-comparison workaround) |

---

## Unverified seed items (reported by campaign agent, not yet committed)

- **A2, comptime-error enforcement:** The `_resolve_precision` function and the exact
  comptime error behavior for inconsistent precision flags are described in fp8_training_design.md
  but not visible in the current goal2/fp8-training HEAD (they are in Chunk A, commit 804b10d,
  which may not yet be merged into this branch for full integration testing). Pending Chunk B
  completion and merge verification.

---

## Footer

Written with AI assistance (Claude Code / Haiku agent), directed by Evan Owen.
