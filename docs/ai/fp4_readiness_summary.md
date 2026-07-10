# FP4 Training Readiness for llm.mojo — Executive Summary

This document synthesizes the FP4 research campaign into a go/no-go decision framework for the maintainer. Two companion documents provide depth: [`fp4_modular_support_research.md`](fp4_modular_support_research.md) covers toolchain and hardware support; [`fp4_training_recipes_research.md`](fp4_training_recipes_research.md) covers stabilization numerics and recipe. **Do not build FP4 without reading the small-model caveats in §4 and the mandatory recipes in §3.**

---

## TL;DR — Go/No-Go Picture

**The verdict: FP4 is reachable, but conditionally.** MAX's own FP4 kernels do not run on GB10 (sm_121 consumer Blackwell; they require sm_100 datacenter hardware and are inference-only regardless). However, NVIDIA's cuBLASLt library ships production sm_120 NVFP4 block-scaled GEMM kernels (both forward and backward) that are almost certainly compatible with sm_121 via a simple 5-line runtime probe. llm.mojo already uses cuBLASLt for bf16 matmuls, so the plumbing exists. **The single open question is whether cuBLASLt dispatches its sm_120 kernels to this sm_121 device; that probe is running now and is the gate to proceeding.** If it passes, the path forward is clear and well-documented by NVIDIA's 2025 research. If it fails, a hand-written Mojo FP4 kernel is feasible but requires more work. **Realistic speedup at 124M scale is ~2× FP8 (not 4× due to quant/dequant overhead), and the loss convergence gap is honest and potentially significant — this is a parity/research exercise, not a throughput win.**

---

## 1. The Technical Path (Pending Probe)

### 1a. Chosen approach: cuBLASLt NVFP4 interop

**Do this first.** Extend `llmm/matmul.mojo`'s existing `cublasLt` bindings to call NVIDIA's sm_120 NVFP4 block-scaled GEMM kernels (present in `libcublasLt.so.12.9.2.10` on this box). The kernels are named `cutlass3x_sm120_bstensorop_…block_scaled_ue4m3xe2m1_…vs16` — NVFP4 format, 16-element blocks with E4M3 per-block scale. Add `CUDA_R_4F_E2M1` operand types and cuBLASLt block-scale descriptors to the existing matmul path.

**Why cuBLASLt:**
- Lowest new surface area; reuses existing GPU-vendor-BLAS integration.
- NVIDIA maintains and tunes these kernels for production use (Nemotron-3, etc.).
- Gets NVFP4 (vs16 blocks) directly, which is more stable for training than MXFP4.
- Supports all three GEMMs (forward, Dgrad, Wgrad) with both quantized and mixed-precision outputs.

### 1b. The one open empirical question (pending probe result)

**Dispatch compatibility: sm_120 → sm_121.**

cuBLASLt's sm_120 NVFP4 kernels are compiled as separate cubins targeting the sm_120 arch. Consumer Blackwell (GB10, sm_121) shares the same 5th-gen Tensor Cores as datacenter Blackwell's consumer variant, and the ISA is compatible — but cuBLASLt's heuristic for selecting which cubin to load is unverified on sm_121.

**What to do:** ~40 lines of C++/CUDA that reuses the `cublasLt` bindings already in `llmm/matmul.mojo`. Allocate tiny e2m1-packed A/B and e4m3 scale tensors, build a cuBLASLt descriptor with `CUDA_R_4F_E2M1` and block-scale attributes, call `cublasLtMatmulAlgoGetHeuristic()` + `cublasLtMatmul()`, and observe whether it returns `CUBLAS_STATUS_SUCCESS` or `CUBLAS_STATUS_NOT_SUPPORTED`. **This probe is in flight.**

**If the probe passes:** Proceed directly to §3's recipe. The path is unblocked.

**If the probe fails:** Fall back to path (b) below — hand-written Mojo FP4 kernel using `float4_e2m1fn` DType and `mma.sync…mxf8f6f4.block_scale…e2m1` PTX intrinsics. The project already has the muscle for this (pre-MAX era); more work, but feasible.

### 1c. Fallback approach (if dispatch fails)

Hand-written Mojo FP4 kernel with `float4_e2m1fn` and inline PTX. Mojo's stdlib and the LLVM backend already support the instruction family; the compiler emits `mma.sync…kind::mxf8f6f4.block_scale…e2m1…` correctly on sm_120/sm_121. This is the "build our own" path the project has done before, now for FP4. Slower to deliver (new kernel from scratch), but gives full control over layout and epilogue.

---

## 2. Mandatory Stabilization Recipe (Mapped to Codebase)

**Do not train in FP4 without these items.** They are load-bearing. Each maps to specific codebase locations or new infrastructure:

### Recipe Checklist

1. **Format: NVFP4, E2M1 elements**
   - 16-element blocks (not 32-element MXFP4).
   - E4M3 (FP8) per-block scale + FP32 per-tensor scale (two-level).
   - Update model layers to declare NVFP4 dtype for eligible GEMMs.
   - **Status:** No existing NVFP4 quantizer; new work.

2. **Stochastic rounding (SR) on gradient tensors** ⚠️ **TODO: CRITICAL**
   - Both Dgrad and Wgrad GEMM operands.
   - Round-to-nearest-even (RNE) for weights and activations (lower variance).
   - **Codebase impact:** `llmm/adamw.mojo` line ~141 has explicit TODO comment: `"TODO: Karpathy adds a stochastic rounding function here for low-precision params."` Currently it's plain truncation (RNE-ish).
   - **Also needed:** Device-side per-element RNG (not present). `llmm/rand.mojo` is host-side MT19937 for weight init only. SR needs a GPU counter-based RNG (Philox / `squares` style).
   - **Leverage:** The FP8 campaign is building shared scaling infrastructure; coordinate on the RNG primitive.

3. **Random Hadamard Transform (RHT), 16×16, on Wgrad only**
   - Fixed random sign vector, shared across all layers and all of training.
   - Fuse into pre-Wgrad-GEMM cast.
   - **Status:** New infrastructure.

4. **2D block scaling for weights; 1D for activations & gradients**
   - Weights: 16×16 blocks (forward/backward consistency).
   - Acts/grads: 1×16 blocks.
   - **Status:** New quantizer + scale-factor layout.

5. **FP32 master weights in the optimizer**
   - Already present in llm.mojo. No change needed.
   - **Status:** ✓ Done.

6. **Selective high-precision layers**
   - Keep ~15% of linear layers in BF16: first ~2 blocks + final several blocks.
   - **Always keep out of FP4:** embeddings, all LayerNorms, LM head, attention QKV/out/softmax.
   - **For 124M:** Apply NVFP4 only to MLP GEMMs of middle blocks.
   - **Status:** Requires per-layer routing logic in the forward pass.

### Summary of New Infrastructure Required

| Item | Codebase Location | Status | Notes |
|------|---|---|---|
| Stochastic-rounding quantizer | `llmm/adamw.mojo:~141` | ❌ Missing | TODO marker exists; blocks FP4 + needed for FP8 |
| Device RNG (Philox/squares) | new file, `llmm/rand_device.mojo` | ❌ Missing | Shared with FP8 pipeline |
| NVFP4 2D/1D quantizer | new quantizer module | ❌ Missing | Format + scale-factor logic |
| 16×16 Random Hadamard | new module | ❌ Missing | Fixed sign vector, pre-GEMM fusion |
| Per-layer precision routing | `llmm/transformer.mojo` or new | ⚠️ Partial | Need BF16 vs FP4 layer selection |
| cuBLASLt FP4 interop | `llmm/matmul.mojo` | ⏳ Pending | Conditional on dispatch probe (§1b) |
| Late-training BF16 switch hook | `train_gpt2.mojo` | ⚠️ Partial | LR-decay trigger; forward-pass only |

---

## 3. Nice-to-Have Stabilization (Recovers the Last Gap)

- **Forward→BF16 switch in final ~18% of training** (at LR-decay onset): recovers ~60% of the residual convergence gap at large scale.
- **Loss-spike / divergence monitoring:** Run a BF16 shadow and track relative validation-loss gap; watch grad-norm vs √3 quantization-noise floor; monitor weight-oscillation rate.
- **Oscillation mitigation & outlier-channel retention** (TetraJet-v2 style): OsciReset (bin re-centering) + keep 5–10% of large-magnitude activation channels in BF16/FP8 late in training. High leverage at 124M scale.

---

## 4. Realistic Expectations at 124M Scale (The Honest Part)

**This is critical context.** Nearly all published FP4 wins (NVIDIA, Quartet, "FP4 All the Way") are at ≥1B params. The anchor 12B result lands within ~1% loss of FP8. GPT-2 124M is **an order of magnitude below** where any paper claims FP4 pays off.

### Why small scale is hard for FP4

- **Gradient-noise floor (√3 threshold):** Once gradient norm < √3 × quantization noise, training stalls. Small models hit this sooner and spend more of training near it.
- **Quartet's scaling law:** FP4 is compute-optimal only at large N/large data; at ~100–300M the quantization-noise penalty typically **outweighs** the compute saving.
- **Fixed overheads don't shrink:** Embeddings, norms, LM head, and attention (all BF16) are a larger FLOP fraction at 124M than at 12B — the FP4-eligible MLP GEMMs are a smaller slice, so the achievable speedup is smaller precisely where the accuracy risk is largest.

### Honest expectation for 124M

- **Convergence gap:** Plausibly a few percent relative loss, possibly a stall near the noise floor. (At 12B with the full recipe, ~1% loss gap; at 124M, expect worse.)
- **Wall-clock benefit:** Little to none on a single GB10, given the quant/dequant overhead and the small-model noise floor.
- **Best-case scenario:** TetraJet-v2 trained OLMo2 70M–370M in NVFP4 with near-FP8 parity (oscillation + outlier tricks). So FP4 *can* converge at our scale — the open question is speedup.

### Framing for success

**Frame this as a parity/research exercise, not a throughput win.** Set the bar as "within X% of BF16 loss at 124M" and measure honestly. The machinery should validate at 124M; defer the "FP4 is faster" claim to larger models where the literature backs it up. If FP4 converges at 124M, that is a genuine achievement (and a validation of NVIDIA's recipe); if it doesn't, you've answered a real research question.

---

## 5. Source Documentation

For implementation depth, recipe details, and citations:

- **[`fp4_modular_support_research.md`](fp4_modular_support_research.md)** — Modular/MAX toolchain support, cuBLASLt verification, the sm_121 dispatch probe, and fallback kernel paths. Read for GPU-vendor plumbing details and the probe specification.
- **[`fp4_training_recipes_research.md`](fp4_training_recipes_research.md)** — NVFP4 format details, stochastic rounding, Random Hadamard Transform, 2D vs 1D scaling, the √3 noise floor, small-model scaling law, and the full checklist with paper citations. Read for numerics and training recipe.

---

Written with AI assistance (Claude Code / Haiku agent), directed by Evan Owen.
