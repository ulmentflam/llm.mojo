# FP8 toolchain capability probe — results

Environment: Mojo `1.0.0b3.dev2026062706` (`5f2bd602`), MAX `26.5.0.dev2026062706`,
NVIDIA GB10 (Grace-Blackwell, aarch64, sm_121). All probes built/run via
`pixi run mojo run -I . <file>` from this worktree (`.pixi` is a symlink to
the main checkout's env). GPU probes wrapped in
`flock -w 10800 /tmp/llmm-gpu.lock -c '...'`.

## Probe 1 — dtype existence, scalar/SIMD casts, arithmetic (host/CPU)

File: `probe1_dtype.mojo`. **PASSES.**

- `DType.float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`
  all exist and print correctly.
- Scalar and `SIMD[DType.float8_e4m3fn/e5m2, N]` casts fp32<->fp8 work on
  the host (CPU) target, including out-of-range saturation:
  - `1000.0 -> e4m3 -> f32 == 448.0` (e4m3 max normal, as expected).
  - `1000.0 -> e5m2 -> f32 == 1024.0` (rounds to nearest representable,
    well within e5m2's ~57344 max).
- **Direct fp8 SIMD arithmetic works on host**: `SIMD[float8_e4m3fn,4] + SIMD[float8_e4m3fn,4]`
  and `*` compile and produce correct results without an explicit
  cast-up/cast-down dance (e.g. `1.0+0.5 -> 1.5`, `3.0*0.5 -> 1.5`).

Conclusion: the dtype and its host-side numeric behavior are fully
implemented and correct; e4m3fn is the standard "fn" (finite, no inf)
variant, e5m2 follows IEEE-ish rules with a larger range.

## Probe 2 — trivial elementwise GPU kernel (fp8 in/out, compute inside)

Files: `probe2_gpu_kernel.mojo` (realistic axpy-shaped kernel),
`probe2a_passthrough_ok.mojo` / `probe2b_arithmetic_fails.mojo` (minimal
bisection). **FAILS to compile — this is the headline finding.**

`probe2_gpu_kernel.mojo` (fp8 load -> cast fp32 -> `x*2+1` -> cast fp8 ->
store) fails GPU codegen:

```
error: failed to lower module to LLVM IR for archive compilation, run LowerToLLVMPipeline failed
oss/modular/mojo/stdlib/std/builtin/simd.mojo:2247:10: error: conversion from 'f8e4m3fn' to 'f32' is not implemented
note: see current operation: %24 = "pop.cast"(%23) : (!kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32>
```

Bisection (`probe2a` vs `probe2b`) isolates the exact trigger:

- **`probe2a_passthrough_ok.mojo`** — load fp8, `.cast[float32]()`,
  immediately `.cast[float8_e4m3fn]()` back, store. **Compiles and runs.**
  The compiler apparently recognizes the cast-up/cast-down round trip as a
  no-op-ish passthrough and never has to materialize a real fp8->fp32
  conversion.
- **`probe2b_arithmetic_fails.mojo`** — identical, but stores `x + 1.0`
  (fp32 output, no down-cast at all) instead of a bare passthrough.
  **Fails** with the same `pop.cast` lowering error as the full kernel.

Additional checks (`probe2c_bf16_cast_also_fails.mojo`, plus two more not
kept as committed files — same bisection technique, same result each time):
- Same failure for `DType.float8_e5m2`, not just `e4m3fn`.
- Same failure for `SIMD[dtype, 4]` vector loads, not just scalar — the
  unlowered op becomes `!kgen.simd<4, f8e4m3fn> -> !kgen.simd<4, f32>`.
- Same failure for a **direct** fp8 -> bf16 cast (skipping the fp32
  intermediate entirely): `probe2c_bf16_cast_also_fails.mojo` gets
  `conversion from 'f8e4m3fn' to 'bf16' is not implemented`. So the gap is
  general across *any* fp8 -> non-fp8 GPU conversion feeding arithmetic,
  not specific to `f32` as the target dtype.

**Conclusion**: on this toolchain, GPU-target fp8_e4m3fn/e5m2 -> {f32, bf16}
conversion codegen is only implemented for a degenerate cast-in/cast-out
passthrough; any real arithmetic on the upcast value hits an unimplemented
LLVM lowering rule (`pop.cast` for `f8eXmY -> {f32,bf16}` on GPU has no
implementation in this Mojo release). **Elementwise fp8 compute kernels
(dequant -> compute -> quant, the standard pattern for e.g. fused
adamw/gelu/layernorm on fp8 buffers, AND the "manual fp8 load -> cast -> bf16"
dequant fallback) are not viable on this GPU with this toolchain build.**
This does not by itself block FP8 *GEMM*, since tensor-core MMA paths (or
vendor libraries called via FFI) consume fp8 register bit patterns directly
rather than through this generic scalar/SIMD cast lowering — see Probes 3-4.

## Probe 3 — FP8 GEMM through MAX's generic `linalg.matmul` prebuilt kernel

File: `probe3_max_gemm_fp8.mojo`. **FAILS to compile.**

`llmm/matmul.mojo`'s vendor-neutral/CPU/Metal path calls
`from linalg.matmul import matmul` — MAX's own generic GEMM dispatcher (used
directly today for bf16/fp32; the fast NVIDIA path instead uses the repo's
own hand-rolled cuBLASLt FFI bindings, see Probe 4). Calling
`matmul[transpose_b=True, target="gpu"](c, a, b, ctx=ctx)` with `a`/`b` as
`float8_e4m3fn` `TileTensor`s (`c` as `float32`), M=N=K=256, fails with the
**exact same** error as Probe 2:

```
error: conversion from 'f8e4m3fn' to 'f32' is not implemented
note: see current operation: %89 = "pop.cast"(%88) : (!kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32>
```

**Conclusion**: MAX's generic/vendor-neutral GEMM kernel internally performs
the same kind of fp8->f32 upcast (for the accumulator and/or output store)
that Probe 2 showed is unimplemented for the GPU target, so this path is
not usable for fp8 on this toolchain build regardless of GPU vendor.

## Probe 4 — cuBLASLt FP8 GEMM (native + fallback)

Files: `probe4_cublaslt_fp8_gemm.mojo` (e4m3-in/e4m3-out — **FAILS at
runtime**), `probe4b_cublaslt_fp8_bf16out.mojo` (e4m3-in/bf16-out — **PASSES**).

Both are self-contained adaptations of `llmm/matmul.mojo`'s own
`_matmul_cublaslt`/`_lt_dt` (which today only maps bf16/fp32/fp16), adding
an `_lt_dt_ext` that maps `float8_e4m3fn`/`float8_e5m2` ->
`DataType.R_8F_E4M3`/`R_8F_E5M2` — the same mapping MAX's own
`linalg.matmul.vendor.blas._convert_to_cublas_datatype` already has (found
by reading `/home/evan/workspace/modular` at the exact commit pinned to
this installed toolchain, `2e6837c494`, "[Release] Pin lockfiles to Mojo
1.0.0b3.dev2026062706, MAX 26.5.0.dev2026062706"). Both use raw cuBLASLt FFI
calls — **no Mojo-level fp8 SIMD arithmetic is involved**, so neither hits
the Probe 2/3 codegen gap; cuBLASLt itself does all the math via an external
call, given only fp8 pointers.

- **`probe4_cublaslt_fp8_gemm.mojo`** (e4m3 x e4m3 -> e4m3, TN format,
  M=N=K=256): compiles and runs the heuristic search fine, but the actual
  `cublasLtMatmul` execution call fails:
  ```
  Unhandled exception caught during execution: failed to operate on CUBLAS due to error: NOT_SUPPORTED
  ```
  This strongly suggests fp8 *output* on this GPU needs an explicit
  `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` attribute (not exercised by this
  probe) rather than fp8 GEMM being unsupported outright — the same
  operands with a non-fp8 output type work (next bullet).
- **`probe4b_cublaslt_fp8_bf16out.mojo`** (e4m3 x e4m3 -> bf16, TN format,
  M=N=K=256): **PASSES.** `max_abs_err ≈ 8.4e-4`, `max_rel_err ≈ 3.1e-3`
  against an fp32 reference computed on the same fp8-quantized operands —
  i.e. the residual error is at bf16 rounding precision, not fp8
  quantization noise. cuBLASLt correctly executes native e4m3 x e4m3 tensor
  inputs through its fp8 tensor-core path on this GB10/sm_121 GPU.

A build-time gotcha found while writing these: cuBLASLt's fp8 TN-format
matrices are **column-major** (`ld=k`), so element (row=k, col=m) of `A`
lives at host offset `m*K+k`, not the row-major `k*M+m` a first draft used —
that bug silently produced a ~100% max-relative-error host reference while
cuBLASLt itself ran fine, i.e. it looked like a GEMM correctness bug but was
actually a reference-computation bug. Documented in both files' comments.

**Conclusion**: the working, low-risk FP8 GEMM path on this toolchain is
**native e4m3 x e4m3 cuBLASLt GEMM with a bf16 (or presumably fp32 — not
directly tested, but fp32 is a strictly "easier" cuBLASLt output type than
bf16) accumulator/output**, TN format, K a multiple of 16. This requires no
"manual load -> cast -> bf16 GEMM" fallback kernel at all (which Probe 2c
showed is itself non-viable on GPU) — cuBLASLt consumes the raw fp8 buffers
directly and produces bf16 output natively.

## Probe 5 — CPU-target hazard check (AArch64 / GB10 host)

Files: `probe5_cpu_target_hazard.mojo` (plain host `vectorize()`, the
`gelu_fwd_cpu` pattern), `probe5b_cpu_target_dispatch.mojo` (the literal
`target: StaticString` comptime `is_cpu[target]()`/`is_gpu[target]()`
dispatch pattern used by `gelu_fwd`/`matmul_fwd`, via
`DeviceContext(api="cpu")`). Both built with `pixi run mojo build` (matching
how the bf16 hazard was originally triggered — an AoT AArch64 build, not
`mojo run`). **Both PASS — no hazard found for fp8, unlike bf16.**

- `probe5_cpu_target_hazard.mojo`: fp8 load -> f32 upcast -> `x*2+1` ->
  fp8 downcast -> store, inside a `vectorize[4, unroll_factor=4]` CPU loop.
  Builds clean, runs, numerics correct.
- `probe5b_cpu_target_dispatch.mojo`: same computation behind a
  `_fp8_op[dtype, target]` function instantiated with `target="cpu"`
  (mirroring `gelu_fwd[dtype, target]`'s `is_cpu[target]()` branch) and
  driven by a real `DeviceContext(api="cpu")`. Builds clean, runs, numerics
  correct.

**Conclusion**: unlike bf16 (which crashes AArch64 codegen when any
`"cpu"`-target kernel is instantiated — see
`bf16-build-needs-gpu-only-dispatch`), fp8 CPU-target codegen on this
GB10/AArch64 box is fine, at least for this scalar-cast-and-arithmetic
pattern with `vectorize`. This makes sense given the split found in Probes
1-3: fp8<->fp32 SIMD cast lowering IS implemented for the CPU target (this
is exactly what Probe 1 already exercised, just via `mojo run` instead of
`mojo build`) — the codegen gap found in Probes 2/3 is specific to the
**GPU** target's lowering pipeline, not fp8 dtype support in general. No
GPU-only comptime guard is *required* for fp8 CPU-side code purely to avoid
a crash (though CPU-target fp8 kernels are moot anyway since Probe 2 showed
GPU fp8 compute doesn't work, so any real fp8 training kernel would need to
run on CPU regardless — this just confirms that fallback path itself
doesn't crash the AArch64 backend).

## Capability matrix (final)

| Probe | Result | Notes |
|---|---|---|
| 1. dtype exist/cast/arith (host) | **WORKS** | `DType.float8_e4m3fn`, `float8_e5m2` (+`fnuz` variants) fully functional on CPU/host, incl. direct SIMD arithmetic and saturation |
| 2. GPU elementwise kernel (compute) | **FAILS** | `pop.cast f8eXmY -> {f32,bf16}` unimplemented for GPU target once the upcast value is used in arithmetic; bare passthrough cast compiles |
| 3. MAX generic `linalg.matmul` w/ fp8 inputs (GPU) | **FAILS** | Same `pop.cast` gap as probe 2 — MAX's vendor-neutral GEMM kernel hits it internally |
| 4a. cuBLASLt fp8 e4m3 x e4m3 -> e4m3 (GPU) | **FAILS (runtime)** | `CUBLAS_STATUS_NOT_SUPPORTED`; likely needs `D_SCALE_POINTER`, not tested further |
| 4b. cuBLASLt fp8 e4m3 x e4m3 -> bf16 (GPU) | **WORKS** | max_rel_err ≈ 3e-3 (bf16-rounding precision); TN format, K%16==0 |
| 5. CPU-target hazard (AArch64) | **NO HAZARD** | Both plain `vectorize()` and `target="cpu"` comptime dispatch build+run clean for fp8 (unlike bf16) |

## Recommended lowest-risk FP8 GEMM path for this box

**Native cuBLASLt e4m3 x e4m3 -> bf16 GEMM**, i.e. extend
`llmm/matmul.mojo`'s `_lt_dt`/`_matmul_cublaslt` to:

1. Map `float8_e4m3fn`/`float8_e5m2` -> `DataType.R_8F_E4M3`/`R_8F_E5M2` in
   `_lt_dt` (currently falls through to `R_16F`, silently wrong for fp8).
2. Force the TN format (`transA=True, transB=False`) whenever the operand
   dtype is fp8 — cuBLASLt requires it, and `_matmul_cublaslt` currently
   takes `transA`/`transB` as caller-supplied params rather than deriving
   them from dtype, so this needs a dtype-conditional override at the call
   site (matmul_fwd/matmul_d_input_bwd/matmul_d_weight_bwd already choose
   transA/transB per-callsite, so this is a per-callsite change, not a
   structural one).
3. Decouple the input dtype from the output/accumulator dtype (today
   `_matmul_cublaslt[dtype]` is single-dtype for A/B/C/D) — output/master
   weights stay bf16 (matching the existing `LLMM_BF16` mixed-precision
   convention: bf16 activations/grads, fp32 master weights/optimizer
   state), with fp8 used purely for the A/B GEMM operands.
4. Do **not** attempt fp8 output from the GEMM (Probe 4a) without first
   sorting out `D_SCALE_POINTER` — treat that as an unexplored, separate
   follow-up, not part of the lowest-risk path.
5. Do **not** attempt any elementwise fp8 quantize/dequantize *inside a GPU
   kernel* (Probes 2/2c) — that codegen path is broken outright on this
   toolchain for both e4m3 and e5m2, to both f32 and bf16 targets. Any
   fp8 <-> higher-precision conversion needed outside the GEMM operands
   themselves (e.g. producing the fp8 activations to feed in) would need to
   happen via **host-side casts** (Probe 1, works) or some other GPU
   mechanism not covered by this probe suite (e.g. a store-only kernel that
   never upcasts-then-computes, matching the one working case found in
   Probe 2a) — this is a real, unresolved gap for a *quantization* step
   (as opposed to the GEMM itself) in any fp8 training pipeline on this box.
