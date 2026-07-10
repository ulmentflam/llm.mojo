# fp32-parity static gap analysis (2026-07-10)

Read-only source analysis of why `llm.mojo` fp32 (417.72 ms/step, B=4 T=1024,
GPT-2 124M, NVIDIA GB10) is 1.40x behind `llm.c` CUDA fp32 (298.53 ms/step),
even though `llm.mojo` bf16 is at step-time parity with `llm.c` CUDA bf16
(135.97 vs 135.77 ms — see
[`ai_assisted_optimizations_and_benchmarks.md`](ai_assisted_optimizations_and_benchmarks.md)
line 1663 for the source benchmark table). No GPU access was used to produce
this document; every claim below is a direct code citation plus arithmetic on
publicly-documented tensor-core ratios. It should be treated as a set of
falsifiable predictions for the next benchmarking pass, not a verified result.

## Bottom line

**TF32 is the dominant, essentially sole, lever.** `llm.mojo`'s fp32 GPU
matmul and attention GEMMs never enable TF32 tensor-core math — they run on
plain FP32 CUDA cores — while `llm.c`'s fp32 arm auto-enables TF32 on any
compute-capability-8.0+ GPU (GB10 qualifies). Every other optimization that
carried bf16 to parity (fused epilogues, coalesced dbias, memset elimination,
LN-backward fusion) is dtype-generic and already applies equally to the fp32
build — static reading of `layernorm.mojo`, `zero.mojo`, `memory.mojo`,
`adamw.mojo`, `global_norm.mojo`, `gelu.mojo`, `crossentropy.mojo`,
`fused_classifier.mojo`, and `softmax.mojo` found no bf16-only comptime gate
on any of those optimizations (see "What's already generic" below).

## Gap 1 (dominant): FC-layer GEMMs hardcode `ComputeType.COMPUTE_32F`

**Ours:** `llmm/matmul.mojo:214`, inside `_matmul_cublaslt` (the single
cuBLASLt entry point used by `matmul_fwd`, `matmul_d_input_bwd`, and
`matmul_d_weight_bwd` — i.e. every FC-layer forward/backward GEMM in the
model):

```mojo
check_cublas_error(
    cublasLtMatmulDescCreate(
        UnsafePointer(to=desc).as_unsafe_any_origin(),
        ComputeType.COMPUTE_32F,     # <-- hardcoded, ignores dtype
        DataType.R_32F,
    )
)
```

`ComputeType.COMPUTE_32F` (value 68, defined in
`_cublas/cublas.mojo:2679`) is plain FP32 math — no tensor cores. The
sibling enum value `ComputeType.COMPUTE_32F_FAST_TF32` (value 77,
`_cublas/cublas.mojo:2683`) is what actually routes FP32-input GEMMs through
TF32 tensor cores. For a **bf16** call, `dt = _lt_dt[dtype]() = R_16BF`
already signals cuBLASLt to use tensor cores regardless of the compute-type
enum (mixed-precision GEMMs are tensor-core-eligible by input dtype alone),
so `COMPUTE_32F` is correct and harmless there — this is exactly why bf16
reached parity without ever touching this line. For **fp32** it silently
caps every FC GEMM (`qkv`, `attproj`, `fc`, `fcproj`, `lm_head`) at
non-tensor-core throughput.

**llm.c's fp32 arm:** `third_party/llm.c/train_gpt2_fp32.cu:1614-1618`:

```c
// TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
```

`cublas_compute_type` is then passed into every `cublasLtMatmulDescCreate`
call in `llmc/matmul.cuh:126`. GB10 is a Blackwell part (compute capability
well above 8.0), so this is unconditionally `CUBLAS_COMPUTE_32F_FAST_TF32` on
our target hardware — **llm.c's "fp32" benchmark arm is TF32-accelerated,
not true IEEE fp32.** This matches the confirming comment already present in
the codebase at `docs/ai/ai_assisted_optimizations_and_benchmarks.md:45`
("llm.c's bf16→fp32 speedup is ~2.2×; PyTorch bf16 ≈ fp32 (its fp32 already
uses TF32...)").

**Fix:** make the compute-type comptime-conditional on `dtype` in
`_matmul_cublaslt`, e.g.

```mojo
comptime compute_type = ComputeType.COMPUTE_32F_FAST_TF32 if dtype == DType.float32 else ComputeType.COMPUTE_32F
```

and pass `compute_type` instead of the literal at line 214. No other change
needed — bf16/fp16 keep their existing (already-correct) behavior.

## Gap 2 (same root cause, attention GEMMs): `attention.mojo:4303`

**Ours:** `llmm/attention.mojo:4266-4306`, the "cuBLAS-symbol-using tail" of
`_attn_gemm_batched` — the single function behind QKᵀ (`_attention_bmm_scoreout`),
A·V (`_attention_bmm_headout`), and the backward dQ/dK/dV batched GEMMs
(`_attention_bmm_kvgrad` and the direct calls at `attention.mojo:4722,4737`).
This path is active for **both** fp32 and bf16 because
`USE_LT_ATTN = False` (`attention.mojo:109`) routes every attention batched
GEMM through the legacy `cublasGemmStridedBatchedEx` API, not through
`_matmul_cublaslt`:

```mojo
cublasGemmStridedBatchedEx(
    ...
    ComputeType.COMPUTE_32F,   # attention.mojo:4303 — hardcoded, same bug
    Algorithm.DEFAULT,
)
```

The `USE_LT_ATTN` comment ("verified NEUTRAL — it selects the identical
cutlass wmma/tensorop kernels as `cublasGemmStridedBatchedEx`") documents a
finding that only holds for the bf16 case it was measured on: for bf16 both
APIs already select tensor-core kernels (input dtype alone triggers this),
so which API is used is irrelevant. That equivalence does **not** hold for
fp32 — `cublasGemmStridedBatchedEx` with `COMPUTE_32F` on FP32 inputs is
plain-CUDA-core math regardless of which of the two cuBLAS APIs carries it,
so this was never actually tested for the fp32 arm.

**llm.c's fp32 arm:** `third_party/llm.c/train_gpt2_fp32.cu:765,776,859,861,868,870`
uses `cublasSgemmStridedBatched` (the legacy typed Sgemm entry point, not
the `Ex` variant) for QKᵀ/A·V forward and all four backward attention GEMMs.
This legacy API has no explicit compute-type argument — it honors the
**handle's** math mode, which was set to `CUBLAS_TF32_TENSOR_OP_MATH` at
line 1618 above. So llm.c's attention path is also TF32-accelerated.

**Fix:** same pattern as Gap 1 — make the `ComputeType` argument at
`attention.mojo:4303` comptime-conditional on `a_dt == DType.float32`. Since
`_attn_gemm_batched` is the single choke point for all four attention GEMM
call sites, one change covers forward and backward.

## Gap 3: none found — the bf16-era optimizations are dtype-generic

Explicitly checked for bf16-only comptime gates (`LLMM_BF16` / `USE_BF16` /
`DType.bfloat16` conditionals) around the specific wins mentioned in the
bf16 campaign, and found none gating the optimization itself:

- **Memset elimination / dead-alloc removal** (`train_gpt2.mojo`,
  `llmm/memory.mojo`): the `enqueue_memset` calls that remain
  (`train_gpt2.mojo:1544,1554,2224,2230,2236`) are all typed
  `Scalar[GPT2_DTYPE]`, i.e. they already resolve to whichever dtype the
  build is compiled for — the elimination logic itself has no dtype branch.
- **LN-backward fusion** (`llmm/layernorm.mojo`): every `dtype`-parameterized
  function (`layernorm_fused_residual_fwd`/`_gpu`/`_cpu`, etc.) is a single
  generic implementation; no `comptime if dtype == DType.bfloat16` branch
  exists anywhere in the file.
- **Selective gradient zeroing** (`llmm/zero.mojo`, `llmm/adamw.mojo`,
  `llmm/global_norm.mojo`): same — grep for `DType.bfloat16`/`USE_BF16`
  across all of `matmul.mojo`, `attention.mojo`, `layernorm.mojo`,
  `fused_classifier.mojo`, `softmax.mojo`, `adamw.mojo`, `memory.mojo`,
  `zero.mojo`, `gelu.mojo`, `global_norm.mojo`, `crossentropy.mojo` turns up
  exactly three hits, all unrelated to compute-path selection: the
  `_lt_dt`/`_cublas_dt` data-type-enum mappers (`matmul.mojo:148`,
  `attention.mojo:3972` — these just pick the cuBLAS `DataType` tag matching
  the buffer's element type, required for correctness, not a perf gate),
  and `USE_FLASH_FWD and dtype == DType.bfloat16` (`attention.mojo:1543`) —
  which is moot because `USE_FLASH_FWD = False` (`attention.mojo:1348`), so
  the flash-attention kernel is disabled for both dtypes currently and isn't
  contributing to the bf16/fp32 gap either way.

**Softmax/attention buffer precision** (task item (c)): the `[T,T]` scores
and probability scratch buffers in `attention_fwd_gemm`
(`attention.mojo:1573-1600`) are typed `dtype` (the function's generic
parameter), so fp32 attention correctly uses fp32-sized score/prob buffers
and bf16 uses bf16-sized ones — this is inherent precision cost present
identically in llm.c's fp32 arm (`llmc/attention.cuh` also keeps
`preatt`/`att` at the model's working dtype), not a llm.mojo-specific gap.

## Theoretical floor: is 298 ms reachable without TF32?

Per-step FLOPs (Kaplan et al. estimator, matching `llmm/mfu.mojo:307-331` and
`llmc/mfu.h`): `flops_per_token = 6N + 6·L·C·T`, N≈124.4M params, L=12,
C=768, T=1024 → 6N≈746.6M, 6·L·C·T≈56.6M → ≈803.2M FLOPs/token.
At B=4,T=1024 → num_tokens=4096 → **≈3.29 TFLOP/step**, close to the
prompt's 3.05 TFLOP order-of-magnitude estimate.

GB10 peaks from `llmm/mfu.mojo:73-74`: **TF32 = 62.5 TFLOPS**,
**BF16(fp32-accum) = 125 TFLOPS** (Blackwell dense ratio BF16:TF32 = 2:1,
per the derivation comment at `mfu.mojo:62-72`). Plain FP32 CUDA-core peak
is *not* in the table (GB10 has no published figure) but on every prior
NVIDIA tensor-core generation the ratio of TF32-tensor-core peak to
plain-FP32-CUDA-core peak is ~4-8x (A100: 156 TFLOPS TF32 vs ~19.5 TFLOPS
FP32 CUDA-core ≈ 8x), so GB10's non-tensor FP32 peak is plausibly in the
≈8-16 TFLOPS range.

Cross-check using our own measured bf16 efficiency as an MFU proxy (matmul
FLOPs dominate the estimator at 93% of the total, so this is a reasonable
transfer): bf16 achieved 3.29 TFLOP / 0.13597 s ≈ 24.2 TFLOP/s against a
125 TFLOPS peak → **MFU ≈ 19.4%**. llm.c's fp32 (TF32) achieved
3.29 TFLOP / 0.29853 s ≈ 11.0 TFLOP/s against a 62.5 TFLOPS peak →
**MFU ≈ 17.6%** — reassuringly close to the bf16 figure, consistent with a
matmul-bound workload hitting a similar MFU fraction independent of which
tensor-core precision is used. Applying that same ≈17.6-19.4% MFU band to
the 62.5 TFLOPS TF32 peak predicts a step time of **≈270-300 ms** if
`llm.mojo` fp32 (after the TF32 fix) reaches an MFU comparable to what it
already reaches in bf16 and what llm.c reaches in fp32 — i.e. **298 ms is
not just reachable, it's the expected outcome, not a stretch target.**

A second cross-check on the *current* (non-TF32) 417.72 ms number: at
3.29 TFLOP / 0.41772 s ≈ 7.87 TFLOP/s achieved. That already sits at
50-100% of the plausible 8-16 TFLOPS plain-FP32-CUDA-core roofline — i.e.
**the current fp32 kernels are not leaving much on the table within the
CUDA-core regime**; the 417 ms figure is close to what a well-tuned SGEMM
should already deliver without tensor cores. This is corroborating evidence
that the 417-vs-298 gap is a *missing hardware-mode* problem (TF32 off),
not a kernel-tuning problem — further CUDA-core-level micro-optimization of
the fp32 matmul/attention kernels has little headroom; enabling TF32 is what
unlocks the next tier.

## Ranked findings

| # | File:line | Gap | Fix | Est. impact |
|---|---|---|---|---|
| 1 | `llmm/matmul.mojo:214` | FC-layer cuBLASLt GEMMs (fwd + both bwd passes) hardcode `ComputeType.COMPUTE_32F`, never `COMPUTE_32F_FAST_TF32`, for fp32 | Make compute-type comptime-conditional on `dtype == DType.float32` | Dominant — this covers ~93% of the model's FLOPs (qkv/attproj/fc/fcproj/lm_head GEMMs) |
| 2 | `llmm/attention.mojo:4303` | Attention QKᵀ/A·V forward + dQ/dK/dV backward batched GEMMs (`cublasGemmStridedBatchedEx`, reached because `USE_LT_ATTN=False`) hardcode `ComputeType.COMPUTE_32F` | Same conditional pattern, keyed on `a_dt == DType.float32` | Secondary — attention GEMMs are ~7% of FLOPs but same fix pattern, single choke point (`_attn_gemm_batched`) |
| 3 | — | No other dtype-gated optimization gaps found (memset elimination, LN-bwd fusion, selective zeroing, softmax/attention buffer dtype are all already dtype-generic) | — | None — ruled out |

**Combined estimate:** fixing gaps 1+2 is expected to bring `llm.mojo` fp32
from 417.72 ms to roughly **270-300 ms/step** (≈120-150 ms saved, a
≈1.4-1.55x speedup), landing at or slightly better than llm.c CUDA fp32's
298.53 ms — i.e. **plausible full parity**, using the same "TF32 is
actually what's being measured" trick llm.c already relies on. This is a
prediction to be confirmed by an actual GPU benchmark pass; it is not
independently more certain than the two cross-checks above (MFU-transfer
from bf16/llm.c, and the CUDA-core-roofline sanity check on the current 417
ms figure), both of which converge on the same ~280-300 ms range.

**Is TF32 the dominant lever?** Yes, explicitly and unambiguously — it is
the only place `llm.mojo`'s fp32 GPU code path differs architecturally from
what `llm.c`'s "fp32" binary actually executes on GB10. Every other
optimization identified in the bf16 campaign (fused epilogues, coalesced
dbias reduction, memset elimination, LN-backward fusion, dead-alloc removal)
was confirmed dtype-generic by static reading and already benefits the fp32
build today.

---

Written with AI assistance (Claude Code / Sonnet agent), directed by Evan Owen.
