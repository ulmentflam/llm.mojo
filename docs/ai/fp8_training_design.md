# FP8 (and FP4-ready) Mixed-Precision Training — Design

Target: GPT-2 training in Mojo (`train_gpt2.mojo` + `llmm/`) on an NVIDIA GB10
(Grace-Blackwell, aarch64, compute capability **sm_121a**, unified memory).
Baseline: bf16 mixed-precision training already at llm.c perf parity
(`-D LLMM_BF16=1`).

This document specifies the scheme, the flag axis, a low-precision layer that is
**dtype-generic** (FP8 is its first instantiation; NVFP4 is designed to slot in
without copying machinery), a chunked work plan with dependencies, and strong
verification gates. It is written so that Sonnet coding agents can execute
individual chunks in parallel with minimal cross-file reading.

## 0. Feasibility verdict (read first)

**Feasible today.** Two parts, with different confidence:

1. **The FP8 training *scheme* (storage layout + delayed per-tensor scaling +
   quantize/dequantize numerics + master-weight interplay) is buildable now,
   regardless of the GEMM probe outcome.** Mojo `DType.float8_e4m3fn` /
   `DType.float8_e5m2` are present in the installed stdlib
   (`.pixi/envs/cuda/lib/mojo/std.mojoc`, verified by decompressing the MPKG —
   176 refs to `float8_e4m3fn`). The compiler targets the full Blackwell range
   through `sm_121a` (this box). So we can quantize, scale, run a GEMM, and
   dequantize with fp8 device buffers today.

2. **The *fast* fp8 tensor-core GEMM on sm_121 is the one probe-gated unknown.**
   llm.mojo does **not** use MAX's `linalg.matmul` dispatcher for the linear
   layers — it calls **cuBLASLt directly** (`llmm/matmul.mojo:_matmul_cublaslt`,
   via `from _cublas.cublaslt import cublasLtMatmul`). cuBLASLt has shipped fp8
   GEMM (E4M3×E4M3→bf16/fp32, COMPUTE_32F) since Ada/Hopper; on Blackwell it is
   expected to work through the driver. This is the **primary route** and it fits
   the existing architecture. The MAX `linalg.mojoc` fp8 kernels
   (`fp8_matmul`, `matmul_dynamic_scaled_fp8`, `blockwise_fp8_matmul`, …) also
   exist but their richest paths are **sm_100/sm_100a-tagged** (datacenter B200);
   web docs report MAX's tcgen05 fp8/fp4 kernels *fail to build on sm_120*
   consumer Blackwell (modular/modular#5707), and no fp8 kernel is explicitly
   tagged sm_121. So MAX-linalg fp8 is the **secondary** route, not the primary.

   **A separate probe agent is compiling an fp8 GEMM on this box.** The design
   below routes all GEMM specifics through one primitive (`lowp_gemm`) whose body
   is chosen at comptime; only **Chunk B** depends on the probe. The three probe
   outcomes and the primitive's implementation for each:

   | Probe result | `lowp_gemm` FP8 body | Perf | Numerics/parity |
   | --- | --- | --- | --- |
   | cuBLASLt fp8 lowers + runs on sm_121 | cuBLASLt E4M3 GEMM + scale pointers | speed win expected | real fp8 |
   | cuBLASLt fp8 rejected, MAX linalg fp8 lowers on sm_121 | MAX `matmul_dynamic_scaled_fp8` | speed depends on codegen | real fp8 |
   | neither lowers on sm_121 | **emulated**: quantize bf16→fp8→bf16, GEMM in bf16 (existing path) | no speed win | validates recipe + unblocks FP4 |

   Only the perf upside is probe-dependent. Even the worst case delivers a
   correct, numerically-faithful fp8 training path and the reusable scaling layer
   FP4 will build on — so the deliverable is never blocked.

**Nothing found makes FP8 infeasible.** The only risk is "no speedup on sm_121 if
both vendor fp8 GEMM routes fail to lower," which the emulated fallback contains.

## 1. Chosen scheme

### 1.1 Storage stays bf16; fp8 is a transient inside the GEMM

The single most important decision: **under low precision, model storage stays
exactly what the working bf16 build uses** — parameters/activations/gradients in
bf16 (`GPT2_DTYPE = bfloat16`), fp32 master weights in the optimizer, fp32
statistics/losses (`StatsDType`). fp8 (later fp4) exists **only as short-lived
device-side quantized copies of the two operands feeding a linear GEMM**, plus
their scale/amax scalars. The GEMM consumes fp8 and writes **bf16** output.

Why this layout (and why it is also the right FP4 layout):

- **Neutralizes landmine #2 (host↔device element-size mismatch).** No host buffer
  changes element size: `losses_host_buf` (fp32), `logits_host_buf` (bf16),
  checkpoint params (bf16), master (fp32) are all untouched. The fp8/fp4 buffers
  are device-only transients that are never read back to a typed host buffer in
  the normal path. (See §4 for the one `get_dtype_size` seam that still must be
  patched defensively.)
- **Keeps LayerNorm, softmax, attention core (QKᵀ / softmax·V), GELU, residual
  adds, embeddings, the LM head, cross-entropy/loss, and the AdamW optimizer
  bit-for-bit identical to the bf16 build** — matching the Transformer-Engine
  recipe, which keeps exactly these ops in high precision. Blast radius is three
  functions in `llmm/matmul.mojo`.
- **No forked forward/backward.** `GPT2.forward` / `GPT2.backward` /
  `GPT2.update` are unchanged; they call `matmul_fwd` / `matmul_bwd`, which pick
  the GEMM body at comptime. (Satisfies coordinator requirement #3.)
- **FP4 reuses it verbatim** — storage still bf16, only the transient operand
  dtype + scaling granularity change.

### 1.2 Formats per tensor class (Transformer-Engine HYBRID)

| Tensor / op | Precision | Rationale |
| --- | --- | --- |
| Forward linear GEMM operands (input activation, weight) | **E4M3** (`float8_e4m3fn`) | more mantissa; forward range fits ±448 with per-tensor scale |
| Backward gradient operand (`d_output` into dgrad & wgrad) | **E5M2** (`float8_e5m2`) | gradients need exponent range; tolerate less mantissa |
| Backward's activation/weight operand | **E4M3** | same tensors as forward |
| GEMM accumulate + output | **fp32 accumulate → bf16 out** | standard; output re-enters the bf16 pipeline |
| LayerNorm mean/rstd, softmax, LSE, losses | **fp32** (`StatsDType`) | numerically safe (`train_gpt2.mojo:358`) |
| Attention core (QKᵀ, softmax, softmax·V), residual adds, GELU nonlinearity | **bf16** | precision-sensitive; TE keeps attention out of fp8 |
| Token/pos embeddings (`encoder`), **LM head GEMM** (`train_gpt2.mojo:2080`, its backward `:2284`) | **bf16** | large/skewed dynamic range; fp8 logits corrupt the loss |
| AdamW master weights + m/v moments | **fp32** | unchanged (`llmm/adamw.mojo:57-61`) |

**FP8 GEMMs = the four per-block linear layers only:** QKV projection
(`train_gpt2.mojo:1936`), attention output projection (`:1993`), MLP fc
(`:2023`, gelu-fused), MLP proj (`:2037`), and their backward counterparts
(`:2465` fc, `:2485` proj, `:2548` attn-proj, `:2593` QKV). The LM head is
**excluded** (stays bf16) in v1 for logit/loss stability; a comptime knob can
opt it in later.

### 1.3 Scaling strategy: delayed per-tensor scaling with amax history

- Each fp8 GEMM operand site keeps a small **amax history ring buffer** and a
  **current scale** on device. The scale used this step is derived from *prior*
  steps' amax (delayed scaling) so no max-reduction gates the current GEMM.
- `scale = fp8_max_representable / (amax / 2^margin)`; store `scale_inv` to
  dequantize the accumulator. `fp8_max` = 448 (E4M3), 57344 (E5M2). `margin = 0`
  default. `amax_history_len = 16`, `amax_compute_algo = max` (TE-style
  defaults; both comptime-configurable).
- During a short **warmup** (first `amax_history_len` steps) use current scaling
  (compute this tensor's amax just-in-time) so the history fills with real
  statistics before we trust delayed scales.
- All scale/amax math is **fp32**. (FP4/NVFP4 replaces this per-tensor scalar with
  a **block** scale tensor in `e4m3` — see §3 seams; the granularity is a
  comptime property of the precision spec, so the call sites don't change.)

## 2. Flag scheme — one precision axis

**Decision: introduce a single ordered axis `LLMM_PRECISION`, keep `LLMM_BF16`
as a back-compat alias.** Justification: fp8 and fp4 (and any future e3m4/mxfp8)
are mutually exclusive storage/compute regimes; a growing set of independent
booleans (`LLMM_BF16`, `LLMM_FP8`, `LLMM_FP4`, …) forces error-prone
"exactly-one-of" comptime assertions and cross-products. One axis is DRY and
extensible (coordinator requirement #2). We preserve `-D LLMM_BF16=1` because the
Makefile, docs, and CI already use it and it is load-bearing.

In `train_gpt2.mojo` (replacing the `USE_BF16`/`GPT2_DTYPE` block at lines 93-95):

```mojo
# Precision axis: "fp32" | "bf16" | "fp8" | "fp4"(future). LLMM_BF16=1 is a
# back-compat alias for LLMM_PRECISION=bf16. GPU-only for the low-precision
# regimes (see landmine #1).
comptime PRECISION = _resolve_precision()  # reads LLMM_PRECISION / LLMM_BF16
comptime LOWP_ENABLED = PRECISION == "fp8" or PRECISION == "fp4"
comptime STORAGE_DTYPE = DType.float32 if PRECISION == "fp32" else DType.bfloat16
comptime GPT2_DTYPE = STORAGE_DTYPE            # keep the existing name/usage
comptime MASTER_DTYPE = DType.float32          # unchanged
comptime USE_BF16 = STORAGE_DTYPE == DType.bfloat16   # master kept iff bf16 storage
comptime SPEC = precision_spec[PRECISION]()    # the dtype-generic spec, §3
```

`_resolve_precision()` maps `is_defined["LLMM_BF16"]()` → `"bf16"` and reads a
`LLMM_PRECISION` string define, defaulting to `"fp32"`; it errors at comptime if
both are set inconsistently. **`USE_BF16` keeps its exact current meaning ("keep
an fp32 master + bf16 storage")**, so `llmm/adamw.mojo`, `zero.mojo`, and the
`has_master` plumbing need **no change** — fp8/fp4 get the same fp32 master path
for free.

Makefile: add `build-fp8` / `build-profile-fp8` / `build-infer-fp8` by cloning
the `-bf16` targets (`Makefile:307-357`) and swapping `-D LLMM_BF16=1` for
`-D LLMM_PRECISION=fp8`. Binaries `build/train_gpt2_fp8`, etc. FP8/FP4 **load the
bf16 checkpoint** (`gpt2_124M_bf16.bin`, `EXPECTED_VERSION = 5`,
`train_gpt2.mojo:887`) because storage is bf16 — no new checkpoint format.

**Landmine #1 (AArch64 codegen crash if any "cpu" target is instantiated under
low precision):** every low-precision code path — quantize kernels, scale
kernels, `lowp_gemm` — must be guarded `comptime if is_gpu[target]()` and must
never be instantiated for `is_cpu[target]()`. The CPU/fp32 path stays the
existing bf16-absent path. `_resolve_precision` must reject
low-precision + CPU target at comptime with a clear error (mirror
`train_gpt2.mojo:3903` "bf16 build supports only the GPU target").

## 3. The dtype-generic low-precision layer (`llmm/lowp.mojo`, new file)

This is the DRY core (coordinator requirement #1). **All** scaling machinery
lives here, comptime-parameterized by a `PrecisionSpec`, so FP4 is a new spec
value, not new code.

```mojo
struct ScalingKind:            # comptime enum
    PerTensor                  # FP8
    Block1D                    # (reserved)
    Block2D                    # NVFP4: 16-elem blocks, 2D tiling

@fieldwise_init
struct PrecisionSpec:
    var fwd_dtype: DType           # e4m3fn (fp8) / e2m1 packed (fp4)
    var bwd_dtype: DType           # e5m2   (fp8) / e2m1        (fp4)
    var scale_dtype: DType         # fp32   (fp8) / e4m3fn      (nvfp4 block scale)
    var scaling: ScalingKind       # PerTensor (fp8) / Block2D (fp4)
    var block: Int                 # 0 (fp8) / 16 (nvfp4)
    var amax_history_len: Int      # 16
    var margin: Int                # 0
    var stochastic_rounding: Bool  # False (fp8) / True (fp4)
    var hadamard: Bool             # False (fp8) / True (fp4, optional)

fn precision_spec[name: StaticString]() -> PrecisionSpec  # "fp8" -> FP8_SPEC, ...
```

Generic functions (all `comptime if is_gpu[target]()` guarded, GPU-only):

- `struct AmaxState[spec]` — device ring buffer of amax history + current
  scale(s) per GEMM operand site. **PerTensor:** one fp32 scale + `history_len`
  fp32 amaxes. **Block2D seam:** a `[rows/block, cols/block]` tensor of
  `scale_dtype` scales; the struct's storage size is a `comptime` function of
  `spec.scaling`.
- `fn compute_amax[spec, in_dtype](x, amax_out, ctx)` — reduction. PerTensor →
  scalar; Block2D → per-block (seam: tiled reduction).
- `fn update_scale[spec](amax_history, scale_out)` — delayed-scaling scale from
  history (`max`/`most_recent`, `margin`). Block seam: per-block scale.
- `fn quantize[spec, out_dtype, in_dtype](x_bf16, x_lowp_out, scale, ctx)` —
  cast bf16→lowp applying `scale`. **Seams:** (a) optional **stochastic
  rounding** at the narrowing cast (`spec.stochastic_rounding`) via the SR cast
  helper from Chunk G — RNE (plain `cast`) otherwise; (b) optional **Hadamard
  transform** of `x_bf16` before quantize (`spec.hadamard`) — the insertion point
  is *here*, first line of quantize; (c) **block packing** for fp4 (2 values/byte)
  lives behind `comptime if spec.block > 0`.
- `fn sr_cast[out_dtype, in_dtype, w](x, rng_state) -> SIMD[out_dtype, w]` and a
  **counter-based device RNG** (`Philox4x32` or `squares`, stateless/counter-keyed
  so it is deterministic and race-free) — the only new device-RNG in the tree
  (`llmm/rand.mojo` is host-only MT19937). Built in **Chunk G**; the seam is
  `spec.stochastic_rounding`. SR rounds a value to a representable low-precision
  neighbor with probability proportional to distance, so repeated
  master→low-precision stores don't systematically truncate — the missing piece
  behind `llmm/adamw.mojo:141`'s TODO.
- `fn dequantize_accum[spec](acc, scale_inv_a, scale_inv_b, out_bf16)` — apply
  the product of inverse operand scales to the GEMM accumulator → bf16.
- `fn lowp_gemm[spec, transA, transB, a_dtype, b_dtype](d_bf16, a_bf16, b_bf16,
  a_state, b_state, m, n, k, accumulate, ctx)` — **the single GEMM entry
  point**: quantize both operands (with their scales), call the vendor fp8/fp4
  GEMM (Chunk B body), dequantize to bf16 `d`. This is the only function whose
  body the probe changes.

**FP8 instantiation:** `FP8_SPEC = PrecisionSpec(e4m3fn, e5m2, fp32, PerTensor,
0, 16, 0, False, False)`.

**FP4/NVFP4 extension seams (documented, not built now):**
1. `FP4_SPEC = PrecisionSpec(e2m1, e2m1, e4m3fn, Block2D, 16, …, stochastic=True,
   hadamard=True)` — no new call sites, only a new spec constant.
2. `ScalingKind.Block2D` branches already carved in `AmaxState`,
   `compute_amax`, `update_scale`, `quantize`.
3. Stochastic-rounding hook = point (a) in `quantize`, provided by Chunk G's
   `sr_cast` + counter-based device RNG (also retires `llmm/adamw.mojo:141`).
4. Hadamard hook = point (b) in `quantize` (spread outliers before 4-bit
   narrowing); its inverse folds into `dequantize_accum`.
5. Block-scale storage dtype = `spec.scale_dtype` (e4m3 for NVFP4).
6. Packing (2×e2m1/byte) behind `comptime if spec.block > 0`; the one
   `get_dtype_size` seam (§4) must special-case sub-byte dtypes.

## 4. Buffers whose element size changes — explicit audit (landmine #2)

Under §1.1 storage-stays-bf16, **no host↔device typed buffer changes size.**
New allocations are device-only:

| Buffer | Where | Dtype | Host readback? |
| --- | --- | --- | --- |
| fp8 quantized operand scratch (2 per GEMM site, reused) | `llmm/lowp.mojo` / GPT2 init | `spec.fwd_dtype` / `spec.bwd_dtype` | no |
| per-site amax history + scale | `AmaxState` in GPT2 | fp32 (fp8) | no (debug only) |
| GEMM accumulator (if not vendor-internal) | `lowp_gemm` | fp32 | no |

**The one defensive patch required:** `llmm/io.mojo:6-18 get_dtype_size` returns
`4` for any unrecognized dtype (`else: return 4`). If an fp8/fp4 buffer ever
reaches this function (a debug dump, a future checkpoint), it silently gets a
4× / 8× wrong byte count — the exact class of landmine #2. **Add explicit cases:
`float8_e4m3fn`/`float8_e5m2`/… → 1 byte; error (or dedicated packed handling)
for sub-byte fp4.** Do this in Chunk A even though the normal path never hits it,
because it is a latent duplicate of the original bf16 readback bug.
`llmm/checkpointing.mojo:139` (`dtype == bfloat16 or float16`) needs no change
while we store bf16.

## 5. GEMM integration points (exact, `llmm/matmul.mojo`)

Three functions gain a `comptime if LOWP_ENABLED and is_gpu[target]()` branch
that calls `lowp_gemm`; the existing bf16/fp32 cuBLASLt path is the `else`. No
call site in `train_gpt2.mojo` changes.

1. **`matmul_fwd` (`:523`)** — forward linear. Today it calls
   `_matmul_cublaslt[dtype, transA=True, transB=False]` (`:669`) with fused
   bias/GELU epilogue (`epi=164/160/4/1`). FP8 path: `lowp_gemm` with E4M3
   operands → bf16 `out`, then apply **bias/GELU as the existing separate bf16
   kernel** (`bias_gelu_fwd`, `:697`; or the `USE_GELU_FUSION=False` split at
   `:632-659` which already writes `pre_gelu` then a standalone GELU) — because
   cuBLASLt fp8 has restricted epilogue support, do bias+GELU in bf16 after the
   fp8 GEMM. Guard so the LM-head call (`has_bias=False`, from
   `train_gpt2.mojo:2080`) stays bf16 (exclude by a comptime `is_lm_head` param
   or by the caller passing `use_lowp=False`).
2. **`matmul_d_input_bwd` (`:1092`)** — dgrad. Today
   `_matmul_cublaslt[transA=False, transB=False]` (`:1200`), optional DGELU
   epilogue (`epi=192`). FP8: `d_output` operand **E5M2**, `weight` operand
   **E4M3** → bf16 `d_input`; DGELU applied as the existing separate bf16 scaling
   kernel (`_launch_matmul_gelu_backward_scaling_gpu`, `:1186`).
3. **`matmul_d_weight_bwd` (`:1390`)** — wgrad. Today
   `_matmul_cublaslt[transA=False, transB=True]` (`:1437`) with
   `accumulate` folding grad-accum via `beta`. FP8: `input` operand **E4M3**,
   `d_output` operand **E5M2** → bf16 `d_weight`. **Grad-accumulation seam:**
   cuBLASLt fp8 D-scaling + `beta`-accumulate interact; if the vendor rejects
   `beta≠0` with fp8, accumulate in a bf16 scratch and add — note this in Chunk E.

cuBLASLt fp8 specifics for Chunk B (from cuBLASLt fp8 API; **verify against the
installed `_cublas` enum + probe**): requires the **TN layout** (`transA=T,
transB=N`) for the classic fp8 GEMM — the forward already uses `transA=True`
(`:669`), the backward two use other combos, so Chunk B must confirm which
operand orientations the installed cuBLASLt fp8 accepts and may need an explicit
transpose for the dgrad/wgrad orientations; scale pointers via
`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `_B_SCALE_POINTER` (and optional
`_D_SCALE_POINTER`, `_AMAX_D_POINTER`); `ComputeType.COMPUTE_32F`; D in bf16;
dims multiples of 16. `_lt_dt` (`:146`) must gain `float8_e4m3fn → R_8F_E4M3`,
`float8_e5m2 → R_8F_E5M2` **iff those enumerators exist in `_cublas.dtype`**
(the `.mojoc` is compressed; the probe / a source-level compile confirms).

## 6. Chunked work plan

Dependency graph:

```
        A (foundation)
       /       |        \
      B        C         G          <- B ∥ C ∥ G
   (GEMM prim) (lowp layer) (device RNG + SR cast)
       \       / \         /
        \     /   \       /
     D (fwd)  E (bwd, uses G's SR on grad quantize)
          \      /
           F (e2e verify)
```

- **A → {B, C, G}**: B, C, G run in parallel (different files/functions).
- **G → {E (optional-for-FP8), and the AdamW TODO}**; G is **required for FP4**.
- **{B, C} → {D, E}**: D and E run in parallel (different functions in
  `matmul.mojo`; coordinate to avoid edit collisions — D owns `matmul_fwd`, E
  owns `matmul_d_input_bwd`+`matmul_d_weight_bwd`).
- **{D, E} → F**.
- **Probe gate:** only **B** depends on the probe. If the probe is not yet back
  when B starts, B implements the **emulated** body first (always compiles/runs),
  exposes the same `lowp_gemm` signature, and swaps in the cuBLASLt/MAX body when
  the probe confirms — D/E/F are unaffected either way.

### Chunk A — Precision axis + foundation (no deps; unblocks all)
Files/functions:
- `train_gpt2.mojo:93-95` → `_resolve_precision`, `PRECISION`, `LOWP_ENABLED`,
  `STORAGE_DTYPE`, `GPT2_DTYPE`, `USE_BF16` alias, `SPEC` (§2). Comptime error on
  low-precision + CPU target.
- `Makefile:307-357` → `build-fp8`, `build-profile-fp8`, `build-infer-fp8`
  (clone `-bf16`, `-D LLMM_PRECISION=fp8`).
- `llmm/io.mojo:6-18 get_dtype_size` → fp8 → 1 byte; fp4/sub-byte guard (§4).
- Stub `llmm/lowp.mojo` with `PrecisionSpec`, `ScalingKind`, `precision_spec`,
  `FP8_SPEC`, and `precision_spec["fp8"]` returning it (no kernels yet).
Gate **A**: `make build-fp8` compiles **and** `make build` (fp32 CPU) still
compiles (proves landmine #1 — no CPU instantiation of low-precision code).
`build/train_gpt2_fp8` runs 20 steps and, because no GEMM is rewired yet,
produces a loss trajectory **bit-identical to `build/train_gpt2_bf16`** (it *is*
the bf16 build with an inert flag). Check binary mtime before trusting the run.

### Chunk B — `lowp_gemm` primitive + quantize/dequantize kernels (deps: A; PROBE-gated)
Files/functions: `llmm/lowp.mojo` (`quantize`, `dequantize_accum`, `lowp_gemm`),
`llmm/matmul.mojo:146 _lt_dt` (+fp8 enumerators), a new
`_matmul_cublaslt_fp8` (or `_matmul_cublaslt` extended with scale pointers,
E4M3/E5M2 operands, bf16 D, no fused epilogue). Implement the emulated body
first, then the probe-selected vendor body (§1, §5).
Gate **B** (unit test, new `tests/test_lowp_gemm.mojo`): multiply two known bf16
matrices via `lowp_gemm` (E4M3×E4M3→bf16) vs a bf16 `matmul` reference. Assert
**per-element relative error < 2⁻³** (E4M3 has ~3 mantissa bits) on well-scaled
inputs, **no NaN/Inf**, correct output dtype/shape. Include an ill-scaled case
(large amax) to prove the scale prevents overflow. Assert the emulated and vendor
bodies agree to fp8 rounding.

### Chunk C — dtype-generic scaling layer (deps: A; ∥ B)
Files/functions: `llmm/lowp.mojo` — `AmaxState[spec]`, `compute_amax`,
`update_scale` (delayed scaling, warmup, `margin`, `amax_history_len`), plus the
`ScalingKind.Block2D` seams stubbed with `constrained`/TODO so the shape is real
but FP4 is not built (§3).
Gate **C** (unit test, `tests/test_lowp_scaling.mojo`): feed a synthetic amax
sequence; assert (1) warmup uses current scaling, (2) post-warmup scale =
`fp8_max / (max(history)/2^margin)` within fp32 tolerance, (3) scales stay finite
and non-zero across a spike, (4) ring buffer wraps correctly, (5) determinism —
same input → identical scale bits on repeat.

### Chunk G — Counter-based device RNG + stochastic-rounding cast (deps: A; ∥ B, C; optional-for-FP8, required-for-FP4)
Motivation: the tree has **no device RNG and no stochastic rounding**
(`llmm/rand.mojo` is host-only MT19937; `llmm/adamw.mojo:141` has a standing TODO
that the master→low-precision store is a plain RNE `cast[dtype]()` at `:142-144`).
FP4 (and, more mildly, e5m2 gradient quantization) wants SR so repeated narrowing
doesn't systematically truncate.
Files/functions:
- `llmm/lowp.mojo` — `Philox4x32` (or `squares`) **counter-based stateless**
  device RNG keyed by `(global_tid, step, site_id)` so it is deterministic and
  race-free (no shared RNG state → preserves Chunk F bit-stability); and
  `sr_cast[out_dtype, in_dtype, w](x, key) -> SIMD[out_dtype, w]` that adds a
  uniform dither in the ULP gap before the narrowing cast.
- Wire into `quantize`'s seam (a) behind `comptime if spec.stochastic_rounding`.
- **Close the AdamW TODO** at `llmm/adamw.mojo:139-144`: replace
  `param.cast[dtype]()` with `sr_cast` **behind a comptime flag**
  (`SPEC.stochastic_rounding`) so bf16/fp32 keep exact current RNE behavior and
  only fp8(optional)/fp4 opt into SR. Thread a per-element key from
  `(idx, t)` — `t` (the AdamW step) is already a kernel arg.
For FP8 this is **non-blocking**: Transformer-Engine ships FP8 without SR and RNE
is fine for e4m3/e5m2 at 124M scale, so FP8 correctness (Chunks D/E/F) does **not**
depend on G. G lands the reusable helper + the seam so FP4 drops in and the AdamW
TODO is retired.
Gate **G** (unit test, `tests/test_sr_cast.mojo`): (1) SR cast is **unbiased** —
over many keys, `E[sr_cast(x)] ≈ x` for x between two representable
low-precision values (assert mean within a tight tolerance); (2) output only ever
takes the two bracketing representable values; (3) **determinism** — same
`(value, key)` → identical result on repeat (counter-based, no state); (4) with
`stochastic_rounding=False` the AdamW store is **bit-identical** to today's RNE
path (guards the bf16/fp32 regression).

### Chunk D — Forward FP8 integration (deps: A, B, C; ∥ E)
Files/functions: `llmm/matmul.mojo:matmul_fwd (:523)` — add
`comptime if LOWP_ENABLED and is_gpu[target]()` → `lowp_gemm` (E4M3) + separate
bf16 bias/GELU; keep LM-head bf16 (§5.1). Wire the four block forward GEMMs’
`AmaxState`s (allocated in GPT2 init, keyed per site). No change in
`train_gpt2.mojo` forward.
Gate **D**: run 1 forward step under fp8 vs bf16 on the same batch/seed; for the
block linear **activation outputs** (post-GEMM, pre-nonlinearity), assert
per-tensor **cosine similarity > 0.999** and **relative L2 < 0.02**; loss on
step 0 within **2%** of the bf16 loss; no NaN/Inf.

### Chunk E — Backward FP8 integration (deps: A, B, C; ∥ D)
Files/functions: `llmm/matmul.mojo:matmul_d_input_bwd (:1092)` (E5M2 d_output ×
E4M3 weight) and `matmul_d_weight_bwd (:1390)` (E4M3 input × E5M2 d_output),
both behind the same comptime branch; DGELU/bias stay bf16 separate kernels;
handle the wgrad grad-accum `beta` seam (§5.3). Gradient (E5M2) quantization may
opt into Chunk G's SR (`spec.stochastic_rounding`) but does not require it for
FP8. LM-head backward
(`train_gpt2.mojo:2284`) stays bf16.
Gate **E** (strong, per landmine #3 — the flat `atol=2.0` that buried three
backward bugs): after 1 full backward, compare **each of the ~148 gradient
tensors** fp8-run vs bf16-run **individually** — per-tensor **cosine similarity
> 0.99** and **relative L2 < 0.1**; also assert the fp8 global grad-norm
(`global_norm_squared`) within 5% of bf16. **No single flat atol over the
concatenated grad vector** (that is exactly the gate that hid bugs). Reuse
`test_gpt2.mojo`’s reference-gradient harness if present.

### Chunk F — End-to-end verification, determinism, perf (deps: D, E)
Files: extend `test_gpt2.mojo` / a training smoke script; record in
`docs/ai/`.
Gate **F** (all must pass):
1. **Loss-trajectory envelope:** run N=50 steps fp8 vs bf16 (same data/seed);
   assert per step `|loss_fp8 − loss_bf16| ≤ 0.02 + 0.01·k` (growing envelope),
   **not** a single endpoint check.
2. **Bit-stability:** two fp8 runs, same seed → **bitwise-identical** loss
   sequence (catches nondeterministic amax/scale races).
3. **NaN/Inf sentinel** on loss and grad-norm every step.
4. **Memory + perf:** record step time and MFU vs bf16; fp8 must not regress
   correctness; log whether the probe-selected GEMM gives a speed win (expected
   with cuBLASLt fp8, neutral under emulation). Compare against the bf16 parity
   baseline in `docs/ai/ai_assisted_optimizations_and_benchmarks.md`.
5. Run via `make` targets (not bare binaries); verify binary mtime first.

## 7. Open items the executing agents must resolve
- **Probe outcome** decides Chunk B's vendor body (cuBLASLt fp8 vs MAX linalg fp8
  vs emulated). Interface is fixed regardless.
- **`_cublas.dtype` fp8 enumerators** (`R_8F_E4M3`/`R_8F_E5M2`) — confirm by a
  source-level compile in Chunk B (the `.mojoc` is compressed; grep can't see it).
- **cuBLASLt fp8 operand-orientation constraints** for the dgrad/wgrad GEMMs
  (TN-only?) — may force an explicit transpose (Chunk E).
- **wgrad `beta`-accumulate with fp8** — bf16-scratch fallback if unsupported.

---

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
