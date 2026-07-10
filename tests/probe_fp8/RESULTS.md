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

Additional checks (not kept as committed files, run interactively during
bisection, same result each time):
- Same failure for `DType.float8_e5m2`, not just `e4m3fn`.
- Same failure for `SIMD[dtype, 4]` vector loads, not just scalar — the
  unlowered op becomes `!kgen.simd<4, f8e4m3fn> -> !kgen.simd<4, f32>`.

**Conclusion**: on this toolchain, GPU-target fp8_e4m3fn/e5m2 -> fp32
conversion codegen is only implemented for a degenerate cast-in/cast-out
passthrough; any real arithmetic on the upcast value hits an unimplemented
LLVM lowering rule (`pop.cast` for `f8eXmY -> f32` on GPU has no
implementation in this Mojo release). **Elementwise fp8 compute kernels
(dequant -> compute -> quant, the standard pattern for e.g. fused
adamw/gelu/layernorm on fp8 buffers) are not viable on this GPU with this
toolchain build.** This does not by itself block FP8 *GEMM*, since tensor-core
MMA paths consume fp8 register bit patterns via hardware intrinsics /
vendor libraries rather than this generic scalar/SIMD cast lowering — see
Probe 3.

## Probe 3 — FP8 GEMM through MAX prebuilt kernels

(filled in below as probe is run)

## Probe 4 — fp8 -> bf16 fallback GEMM

(filled in below as probe is run)

## Probe 5 — CPU-target hazard check

(filled in below as probe is run)

## Capability matrix (running)

| Probe | Result | Notes |
|---|---|---|
| 1. dtype exist/cast/arith (host) | **WORKS** | `float8_e4m3fn`, `float8_e5m2` (+`fnuz` variants) fully functional on CPU/host, incl. SIMD arithmetic |
| 2. GPU elementwise kernel (compute) | **FAILS** | `pop.cast f8eXmY -> f32` unimplemented for GPU target once the upcast value is used in arithmetic; bare passthrough cast compiles |
| 3. MAX GEMM w/ fp8 inputs | TBD | |
| 4. fp8->bf16 fallback GEMM | TBD | |
| 5. CPU-target hazard | TBD | |
