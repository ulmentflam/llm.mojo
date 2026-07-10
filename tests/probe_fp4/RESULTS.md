# FP4 (NVFP4) cuBLASLt dispatch probe — results

Answers the one open question in `docs/ai/fp4_modular_support_research.md` §5:
does the installed `libcublasLt.so.12.9.2.10`'s **sm_120** NVFP4 block-scaled
GEMM cubin (`cutlass3x_sm120_bstensorop_…_ue4m3xe2m1_…_vs16`) dispatch on this
box's **sm_121** GB10?

## Verdict: DISPATCHES

`cublasLtMatmul` with `CUDA_R_4F_E2M1` A/B operands, `CUBLASLT_MATMUL_MATRIX_
SCALE_VEC16_UE4M3` block scaling (NVFP4, block=16), `f32` accumulate, `bf16`
output returns `CUBLAS_STATUS_SUCCESS` end-to-end on the GB10 (sm_121), and
`nsys profile` shows the GPU actually executed:

```
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

— the exact kernel-name pattern identified by static analysis in the research
doc (§1d), running for real on sm_121 hardware. This upgrades that doc's
**INFERRED** sm_120→sm_121 dispatch hop to **CONFIRMED**.

## How to reproduce

```sh
tests/probe_fp4/build.sh   # nvcc, links against pixi's cudart/cuBLASLt 12.9.2.10
flock -w 10800 /tmp/llmm-gpu.lock -c tests/probe_fp4/probe_fp4
```

`build.sh` compiles with the system CUDA 13.0 toolkit's `nvcc`/headers
(the pixi env ships no headers, only `.so`s) but links directly against the
pixi-pinned `libcublasLt.so.12.9.2.10` / `libcudart.so.12.9.79` via
`-Xlinker <versioned .so path>` + `-rpath` — no symlinks, no system-CUDA
cuBLASLt (13.1.1.3) involved. `ldd probe_fp4` confirms the resolved libs are
the pixi ones.

## What the probe does

512×512×512 NVFP4 GEMM: `D(bf16) = A(e2m1,K-major,opT) @ B(e2m1,K-major,opN)`,
per-16-element-block `ue4m3` scales on A and B, `f32` accumulate. Reference
is an fp32 GEMM over the *unquantized* random inputs (uniform [-3,3]).
Correctness target: relative L2 error < 0.20 (NVFP4 e2m1 has an inherent
~6–12% per-element quantization step at this data range; e4m3 block scales
add negligible extra error).

Host-side quantization (own from-scratch implementation, not borrowed from
any library binary):
- **e2m1 data**: OCP MX 4-bit table `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` × sign,
  nearest-value encode. 2 values packed per byte, even index → low nibble,
  odd → high nibble (convention cross-checked against PyTorch's
  `pack_uint4` in `torch/testing/_internal/common_quantized.py`).
- **e4m3 block scale**: standard FP8 E4M3 (`CUDA_R_8F_UE4M3` is byte-identical
  to `CUDA_R_8F_E4M3` per `library_types.h`), 1 scale per 16-element block
  along K, `scale = block_amax / 6.0`.
- **Scale swizzle**: cuBLAS's 128×4-tile / 32×4×4-internal block-scale-factor
  layout (docs §3.1.4.3.2), implemented from the formula in that section and
  cross-checked against PyTorch's reference `to_blocked()`/`from_blocked()`
  (`torch/testing/_internal/common_quantized.py`, itself citing the same
  cuBLAS doc section) — both independently derived implementations agree.

## Exact API status codes (all `CUBLAS_STATUS_SUCCESS`)

```
cublasLtCreate: CUBLAS_STATUS_SUCCESS
cublasLtMatmulDescCreate: CUBLAS_STATUS_SUCCESS
set A_SCALE_MODE=VEC16_UE4M3: CUBLAS_STATUS_SUCCESS
set B_SCALE_MODE=VEC16_UE4M3: CUBLAS_STATUS_SUCCESS
cublasLtMatrixLayoutCreate(A, CUDA_R_4F_E2M1): CUBLAS_STATUS_SUCCESS
cublasLtMatrixLayoutCreate(B, CUDA_R_4F_E2M1): CUBLAS_STATUS_SUCCESS
cublasLtMatrixLayoutCreate(D, CUDA_R_16BF): CUBLAS_STATUS_SUCCESS
cublasLtMatmulAlgoGetHeuristic: CUBLAS_STATUS_SUCCESS, returned=4
cublasLtMatmul: CUBLAS_STATUS_SUCCESS
cudaEventSynchronize after matmul: no error
cudaGetLastError after matmul: no error
```

No `CUBLAS_STATUS_NOT_SUPPORTED`, `CUBLAS_STATUS_ARCH_MISMATCH`, or any CUDA
runtime error was ever observed, across repeated runs.

## Heuristic results

`cublasLtMatmulAlgoGetHeuristic` returns **4** viable algorithms (asked for
up to 4; all 4 report `workspaceSize=0`, `wavesCount=1.0`) — a non-empty,
non-degenerate result set, i.e. cuBLASLt's own kernel selection considers
this a normal, supported problem shape on this device, not a fallback/error
path.

## Numeric check

- Pure-software dequant GEMM (host-side reimplementation of the same
  packing/scale/decode, no cuBLASLt, no swizzle — isolates "is the quant
  math itself sane"): **rel L2 = 0.1445** vs fp32 reference.
- cuBLASLt NVFP4 GEMM output vs fp32 reference: **rel L2 = 0.1445**
  (matches the software reference to 4 decimal places) — **PASS**.
- Diagnostic: an equivalent bf16 GEMM run through the identical TN/K-major
  layout convention (no FP4 involved at all) reproduces the fp32 reference
  to rel L2 = 0.0029 (bf16 rounding only), confirming the layout/transpose
  scheme itself (not just the FP4 path) is correct.

(First-pass version of this probe had a column-major-vs-row-major output
indexing bug in the comparison code — D is written column-major by cuBLASLt
— which produced a spurious rel L2 ≈ 1.4 on *both* the FP4 and bf16 paths
identically; fixing the readback indexing resolved both simultaneously,
confirming it was a comparison-code bug, not a kernel dispatch/numerics bug.)

## Timing (informational, not the primary deliverable)

At this size (512³) `cublasLtMatmulAlgoGetHeuristic` picked an **old sm_80
WMMA kernel** (`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_
align8`) for the bf16 comparison GEMM rather than a modern sm_120/sm_121
kernel — 512³ is too small for cuBLASLt's heuristic to prefer a bigger,
newer tile config. FP4 GEMM: ~0.044 ms; bf16 comparison GEMM: ~0.18–0.82 ms
(run-to-run variance, other tenants share this GPU) — FP4 measured
**4–8× faster** than this particular bf16 kernel choice at this tiny size.
This ratio is **not a reliable FP4:BF16 throughput estimate** (the bf16
kernel chosen isn't cuBLASLt's best for this hardware/size, and 512³ is far
below the shape where either kernel is compute-bound) — it only further
corroborates that the FP4 path is running a real, fast tensor-core kernel
rather than a slow/emulated fallback. See
`docs/ai/fp4_modular_support_research.md` §3 for the MAMF-based throughput
estimates (~400–500 TFLOPS dense FP4 expected), which remain the throughput
reference; a proper FP4-vs-BF16 throughput comparison needs larger,
compute-bound shapes and is out of scope for this dispatch probe.

## Toolchain

- Device: NVIDIA GB10, `sm_121`.
- Linked libs (confirmed via `ldd`): `libcublasLt.so.12` →
  `.../pixi/envs/cuda/targets/sbsa-linux/lib/libcublasLt.so.12.9.2.10`;
  `libcudart.so.12` → `…/libcudart.so.12.9.79` (both from the pixi env, same
  version pinned as VERIFIED-LOCAL in the research doc).
- Compiler: system `nvcc` (CUDA 13.0.88), `-arch=sm_121`.
- Driver: 595.71.05 (CUDA 13.2 reported by `nvidia-smi`).
- Headers: system CUDA 13.0 `cublasLt.h`/`cuda_runtime.h` (pixi env ships no
  headers; the enums/attrs used — `CUDA_R_4F_E2M1`,
  `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, the `*_SCALE_MODE`/
  `*_SCALE_POINTER` descriptor attributes — are present verbatim and
  numerically stable between the two versions; confirmed by successful
  linkage/execution against the pixi 12.9.2.10 binary).

## Read on next steps

Since the answer is **DISPATCHES**, fallback path (b)(2) in the research doc
(hand-written Mojo `mma.sync` FP4 kernel) is **not** the next move. The
research doc's recommended path (b)(1) — cuBLASLt FP4 GEMM interop via
`llmm/matmul.mojo`'s existing `_cublas.cublaslt` bindings — is validated and
should proceed directly: add `CUDA_R_4F_E2M1` operand support + the block-
scale descriptor attributes (`A_SCALE_MODE`/`A_SCALE_POINTER`, same for B) to
`_matmul_cublaslt`, following the exact attribute sequence exercised here.
No driver/toolkit update is needed for dispatch; the toolchain already
installed (cuBLASLt 12.9.2.10) is sufficient.
