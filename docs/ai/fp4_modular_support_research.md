# FP4 on GB10 via Mojo + MAX — support status (research)

Scope: **can llm.mojo run FP4 matmuls on this box (GB10 / DGX Spark, aarch64,
sm_121, Grace-Blackwell, unified memory) from Mojo/MAX today, and if not, what is
the nearest path?** Companion note `fp4_training_recipes_research.md` covers the
numerics/recipe side; this note covers Modular + hardware *support*.

Toolchain pinned for every VERIFIED-LOCAL claim below:
`mojo 1.0.0b3.*`, `max 26.5.0.dev2026062706`, CUDA 12.9, cuBLASLt 12.9.2.10,
under `/home/evan/workspace/llm.mojo/.pixi/envs/cuda`.

Each claim is tagged **VERIFIED-LOCAL** (found in `.pixi`), **VERIFIED-WEB**
(cited), or **INFERRED**.

---

## TL;DR verdict

- **MAX's own FP4 matmul kernels do NOT run on this GB10.** They are the
  `sm100_structured` block-scaled kernels built on the `tcgen05.mma` instruction,
  which exists only on **datacenter** Blackwell (sm_100a / sm_101a). GB10 is
  **sm_121** (consumer-class Blackwell, warp-level `mma.sync` only, no Tensor
  Memory, no `tcgen05`). MAX has no sm_120/sm_121 FP4 kernel and no CUDA-core
  fallback; the build/dispatch simply fails. (VERIFIED-LOCAL + VERIFIED-WEB)
- **MAX's FP4 is also inference-only** — weight-quantized forward GEMM
  (`x @ Wᵀ`, W packed uint8 `e2m1x2` + block scales). There is **no FP4
  training/backward GEMM** anywhere in the MAX stack. (VERIFIED-LOCAL)
- **The realistic near-term path is cuBLASLt interop**, which llm.mojo *already
  uses* for the bf16 matmul. cuBLASLt 12.9.2.10 on this box ships **sm_120
  NVFP4 block-scaled GEMM kernels** (`cutlass3x_sm120_bstensorop_…block_scaled_
  ue4m3xe2m1_…_vs16`, i.e. e2m1 data, e4m3 scales, block-16 = NVFP4). Whether
  cuBLASLt dispatches these sm_120 cubins onto an sm_121 device is the one open
  question — needs a 5-line runtime probe. (VERIFIED-LOCAL + INFERRED)
- **A hand-written Mojo FP4 kernel is also feasible**: Mojo stdlib has the
  `float4_e2m1fn` DType and the compiler backend already emits the sm_120/sm_121
  `mma.sync…kind::mxf8f6f4.block_scale…e2m1…` PTX. This is the same "write our own
  Mojo kernel" muscle the project already used pre-MAX, now for FP4. (VERIFIED-LOCAL)

So: **no, not out-of-the-box from MAX; yes, very likely reachable via the
cuBLASLt path the project already owns** — pending a dispatch probe on sm_121.

---

## 1. Ground truth on this box (VERIFIED-LOCAL)

### 1a. Mojo/MAX knows FP4 as a type

- `DType.float4_e2m1fn` exists (OCP MX 4-bit e2m1: 1 sign / 2 exp / 1 mantissa,
  bias 1). Found in `max/dtype/dtype.py` (`DType.float4_e2m1fn: "f4e2m1fn"`) and
  interned in `libmax.so`, `libMojoLLDB.so`. The stdlib SIMD scalar is
  `Float4_e2m1fn`. (Also VERIFIED-WEB: docs.modular.com DType reference.)

### 1b. MAX has FP4 matmul ops — but they are inference, block-scaled, sm_100

From `.pixi/…/max/nn/kernels.py` (graph-level ops, backed by Mojo `linalg`
kernels) and `max/nn/quant_ops.py` / `quant_config.py`:

- FP4 GEMM ops present:
  `dynamic_block_scaled_matmul`, `dynamic_block_scaled_matmul_mxfp4`,
  `grouped_dynamic_scaled_mxfp4_matmul`, `grouped_matmul_block_scaled`,
  `_fused_qkv_ragged_matmul_scaled_float4`, plus quant/dequant helpers
  (`quantize_dynamic_block_scaled(_mxfp4)`, `mxfp4_dequant`,
  `mxfp4_preshuffle_*`).
- `QuantFormat.NVFP4` and `QuantFormat.MXFP4` are defined; NVFP4 weight is
  `uint8` (`float4-e2m1x2`) + block scales, activations quantized dynamically.
- The code comments tie the fast path to datacenter Blackwell:
  `quant_config.py` — *"TCGEN-interleaved layout expected by the FP4 matmul
  kernel (NVFP4 only)"*, and MXFP8's sibling notes *"Uses the SM100 block-scaled
  matmul."* **TCGEN = tcgen05 = sm_100.**

These are **weight-quantized inference** matmuls (`x @ Wᵀ`). There is **no
backward/gradient FP4 GEMM** and no FP4 training entrypoint anywhere in the
package. (VERIFIED-LOCAL)

### 1c. The instruction wall: tcgen05 vs GB10

- `libmax.so` interns all Blackwell arch strings — `sm_100/101/103/110/120/121`
  and their `a`/`f` variants — so the compiler *backend* can target sm_121.
  It also contains ~981 `tcgen05` references and the full FP4 PTX mnemonic
  family (`e2m1.e2m1.f32.ue4m3`, `…ue8m0`, `e2m1.e2m3.*`, etc.). (VERIFIED-LOCAL)
- But the **FP4/FP8 structured matmul kernels are compiled against tcgen05**,
  which physically does not exist on sm_121. Presence of the sm_121 arch string
  ≠ presence of an sm_121 FP4 kernel. (VERIFIED-LOCAL + INFERRED)

### 1d. cuBLASLt on this box already ships consumer-Blackwell FP4 GEMM

`libcublasLt.so.12.9.2.10` (`targets/sbsa-linux/lib`) contains **16 sm_120
NVFP4 block-scaled GEMM kernels**, e.g.:

```
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_
  f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_gelu
```

Decoded: `e2m1` A/B data, `ue4m3` (FP8 e4m3) scales, **`vs16`** = scale-vector-16
⇒ **NVFP4** granularity; f32 accumulate; bf16/f16/f32 output; gelu/relu +
bias epilogues; stream-K variants. There are **72 sm_120 cutlass kernels total**
and **1192 sm_100** — i.e. consumer Blackwell is a real, if thinner, target here.
`__nv_fp4_e2m1` conversion symbols are present. (VERIFIED-LOCAL)

- Caveat: **no `cutlass3x_sm121_…` FP4 kernel names** were found — only a
  handful of bare `sm_121` arch-string hits. Consumer-Blackwell FP4 in this
  cuBLASLt is packaged as **sm_120**. Whether the cuBLASLt heuristic serves an
  sm_120 cubin to an sm_121 device is **INFERRED** (see §5, "the one probe").
- llm.mojo already binds cuBLASLt directly: `llmm/matmul.mojo` imports
  `from linalg.matmul.vendor import blas` and `from _cublas.cublaslt import
  cublasLtMatmul, cublasLtMatmulDesc_t, …`. **The interop plumbing exists.**
  (VERIFIED-LOCAL)

---

## 2. Web: Modular's FP4 support status

- MAX changelog documents FP4 **only on SM100 (datacenter B200/GB200)**:
  v26.4 "Added NVFP4 quantization support (Gemma 4)", "MXFP4 for MiniMax-M2";
  v26.3 "NVFP4 grouped matmul … on B200", "MXFP4/MXFP8 block-scaled matmul on
  SM100". **No** mention of sm_120/sm_121, consumer Blackwell, RTX 50, DGX Spark,
  or GB10. (VERIFIED-WEB — docs.modular.com/max/changelog)
- **modular/modular#5707** ("MAX fails to build models on Blackwell sm_120
  because tcgen05 instruction not supported"): **open**, unresolved as of
  2025-12-24. Error text: *"The tcgen05 instructions are only applicable on
  nVidia Blackwell (sm_100a, sm_101a) hardware."* Affects the **FP8 and FP4
  matmul kernels**. Reporter explicitly asks for "a fallback to a naive fp8
  matmul kernel for sm_120"; no Modular-staff fix/roadmap posted. GB10 (sm_121)
  is the same tcgen05-less family. (VERIFIED-WEB)
- **NVIDIA/cutlass#2800**: the Python-DSL `BlockScaledMmaOp` lists only
  `sm_100a` in `admissible_archs`, blocking FP4 on sm_120/sm_121 *at the DSL
  layer* even though the C++/hardware supports it. The issue confirms *"GB10 has
  5th-gen Tensor Cores with FP4 support (1 PFLOPS peak)"* and that the CUTLASS
  changelog lists *"Support for Blackwell SM121 kernels for DGX Spark GPUs"* —
  but notes *"pre-compiled sm_121 kernels may still be unavailable."*
  (VERIFIED-WEB)

Reading: **Modular's FP4 investment is squarely datacenter-Blackwell (sm_100)**
and inference-serving-shaped. Consumer/GB10 FP4 in MAX is not yet a thing, with
no public commitment. Movement will more plausibly arrive first through the
CUTLASS/cuBLAS sm_120/sm_121 kernels than through hand-written MAX tcgen05 code.

---

## 3. Hardware truth — what FP4 can GB10 (sm_121) actually do?

- GB10 has **5th-gen Tensor Cores with hardware FP4**; NVIDIA markets **"1
  PetaFLOP FP4"**, which is the **sparse** peak ⇒ **~500 TFLOPS dense FP4**.
  (VERIFIED-WEB — NVIDIA / CUTLASS#2800)
- **Instruction path is warp-level `mma.sync`, not tcgen05.** For consumer
  Blackwell (sm_120, and sm_121 shares its code), FP4 goes through
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.…e2m1.
  e2m1.f32.ue8m0` — *"No Tensor Memory, no tcgen05."* Note the SM120 report saw
  only **`scale_vec::1X`** (one scale / 32 elems, MXFP4-shaped) compiling, with
  `2X` (NVFP4 block-16) failing to compile — yet this box's **cuBLASLt sm_120
  kernels are `vs16` (NVFP4)**, so NVFP4 *is* reachable through the vendor
  library even if raw hand-written 2X mma.sync was finicky at that report's date.
  (VERIFIED-WEB florianmattana + VERIFIED-LOCAL cuBLASLt)
- **Measured GB10 throughput (MAMF):** ~99.8 TFLOPS BF16, ~207.7 TFLOPS FP8.
  FP8/BF16 ≈ 2.05× — matches the "each step down doubles" tensor-core ladder.
  Dense FP4 should be ~2× FP8 (~400–500 TFLOPS). (VERIFIED-WEB)
- **The project's `llmm/mfu.mojo` ratio assumption checks out.** It encodes
  FP4 : FP8 : BF16 : TF32 = **8 : 4 : 2 : 1** dense (500 / 125 / 62.5 TFLOPS from
  the 1-PFLOP-sparse anchor). The measured FP8:BF16 = 2.05× confirms the 4:2
  rungs; MAMF BF16 ≈ 100 is the *achievable* number vs the 125 *theoretical*
  peak in mfu.mojo (expected — MAMF < spec). **No change needed to the ratio;**
  MFU will read a touch low against the optimistic 125 peak, which is the
  intended conservative behavior. (VERIFIED-LOCAL + VERIFIED-WEB)

---

## 4. Answering the three deliverables

**(a) Can we run FP4 GEMMs from Mojo/MAX on this GB10 today, and by which API?**

- Via **MAX's FP4 kernels**: **No.** They are sm_100 `tcgen05` block-scaled
  kernels; sm_121 has no tcgen05 and MAX ships no sm_120/121 FP4 kernel or
  fallback (modular#5707, open). And MAX FP4 is inference-only regardless — no
  training/backward GEMM to reuse.
- Via **cuBLASLt** (the vendor path llm.mojo already calls for bf16): **very
  likely yes for the forward/backward GEMMs**, using the sm_120 NVFP4
  block-scaled `cutlass3x_sm120_bstensorop_…vs16` kernels present in the
  installed `libcublasLt.so.12.9.2.10` — *if* they dispatch on sm_121. That
  dispatch is the single unverified hop (§5).

**(b) If not out-of-the-box, the nearest path (ranked):**

1. **cuBLASLt FP4 GEMM interop** — reuse `llmm/matmul.mojo`'s existing
   `_cublas.cublaslt` bindings; add `CUDA_R_4F_E2M1` operand types + a
   block-scale descriptor (`CUBLASLT_MATMUL_DESC_*SCALE*`). Lowest new surface,
   uses NVIDIA's tuned kernels, gets NVFP4 (vs16). **Start here.**
2. **Hand-written Mojo FP4 kernel** — `float4_e2m1fn` DType + emit
   `mma.sync…mxf8f6f4.block_scale…e2m1` (inline PTX / target intrinsic); the
   backend already speaks this ISA. More work, full control, matches the
   project's pre-MAX "write our own kernel" pattern. Fallback if cuBLASLt won't
   dispatch on sm_121 or lacks the exact epilogue/layout needed for training.
3. **Wait for MAX** — not advisable: no public sm_120/sm_121 FP4 roadmap, and
   even if it lands it targets *inference weight-quant*, not training GEMMs.

**(c) Throughput expectations:** dense FP4 ≈ **~400–500 TFLOPS** on GB10
(~2× FP8's measured ~208, ~4–5× BF16's measured ~100). Real training MFU will be
well under peak (memory movement, quant/dequant, scale bookkeeping, small GPT-2
shapes) — treat FP4's win as ~1.5–2× over the bf16 baseline in practice, not the
4× ceiling. Block-scaling overhead and the sm_121-vs-sm_120 kernel-quality gap
are the risks.

**(d) Key citations:** see below.

---

## 5. The one probe before committing to the cuBLASLt path

Confirm cuBLASLt serves its **sm_120** NVFP4 kernel to this **sm_121** device:
allocate tiny e2m1-packed A/B + e4m3 scale tensors, build an FP4
`cublasLtMatmulDesc` (`CUDA_R_4F_E2M1`, block-scale attrs), call
`cublasLtMatmulAlgoGetHeuristic` + `cublasLtMatmul` on the GB10, and check for a
non-empty heuristic / `CUBLAS_STATUS_SUCCESS`. If it returns
`CUBLAS_STATUS_NOT_SUPPORTED`, fall to path (b)(2). This is ~40 lines reusing the
bindings already in `llmm/matmul.mojo`.

---

## 6. Empirical dispatch verdict (probe result, 2026-07-10)

**§5's INFERRED sm_120→sm_121 dispatch hop is now CONFIRMED**, by direct
execution — see `tests/probe_fp4/` for the probe and full writeup
(`tests/probe_fp4/RESULTS.md`).

A minimal `cublasLtMatmul` call — `CUDA_R_4F_E2M1` A/B operands,
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` block scaling (NVFP4, block=16),
`f32` accumulate, `bf16` output, 512×512×512 — returns
`CUBLAS_STATUS_SUCCESS` end-to-end against this box's pixi-pinned
`libcublasLt.so.12.9.2.10` on the GB10 (sm_121). `cublasLtMatmulAlgoGetHeuristic`
returns 4 viable algorithms (not empty/degenerate). `nsys profile` confirms
the GPU executed exactly the kernel identified by static analysis in §1d:

```
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

i.e. the **sm_120-named cubin runs, unmodified, on sm_121 hardware** — no
`CUBLAS_STATUS_ARCH_MISMATCH`, no silent CPU/emulated fallback. Numeric
correctness: the GEMM output matches an independent host-side
software-dequant reference to 4 decimal places (rel L2 = 0.1445 both sides),
and a same-layout bf16 control GEMM matches the fp32 reference to rel L2 =
0.0029, confirming the harness itself (transpose/layout convention,
block-scale swizzle) is correct, not just "it didn't crash."

**Revised path ranking (§4b):** path (1), cuBLASLt FP4 GEMM interop via
`llmm/matmul.mojo`'s existing bindings, is validated and is the move — add
`CUDA_R_4F_E2M1` layouts + the `*_SCALE_MODE`/`*_SCALE_POINTER` descriptor
attributes to `_matmul_cublaslt`. Path (2) (hand-written Mojo `mma.sync` FP4
kernel) is not needed for dispatch; no driver/toolkit update is required
either — the already-installed cuBLASLt 12.9.2.10 dispatches correctly today.

---

## Key citations

- Local: `.pixi/…/max/nn/kernels.py`, `max/nn/quant_ops.py`,
  `max/nn/quant_config.py`, `max/dtype/dtype.py`, `libmax.so`,
  `libcublasLt.so.12.9.2.10`, `llmm/matmul.mojo`, `llmm/mfu.mojo` (this repo).
- modular#5707 — MAX build fails on sm_120, tcgen05 only sm_100a/101a:
  <https://github.com/modular/modular/issues/5707>
- NVIDIA/cutlass#2800 — BlockScaledMmaOp FP4 restricted to sm_100a; GB10 FP4 /
  SM121 notes: <https://github.com/NVIDIA/cutlass/issues/2800>
- MAX changelog (NVFP4/MXFP4 on SM100 only):
  <https://docs.modular.com/max/changelog/>
- Mojo DType `float4_e2m1fn`:
  <https://docs.modular.com/mojo/stdlib/builtin/dtype/DType/>
- MAX blockwise_fp8 SM100 structured kernel (tcgen05 family):
  <https://docs.modular.com/max/api/kernels/linalg/matmul/gpu/sm100_structured/blockwise_fp8/blockwise_fp8_matmul_kernel/>
- SM120 FP4 uses warp-level mma.sync, not tcgen05 (throughput + PTX mnemonic):
  <https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/>
- GB10 measured MAMF (BF16 ~99.8, FP8 ~207.7) & 1-PFLOP-FP4-sparse framing:
  <https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993>

---

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
