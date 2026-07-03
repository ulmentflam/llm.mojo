# Apple Silicon (Metal GPU) port: gotchas and optimizations

A technical log of the July 2026 campaign to port and optimize `llm.mojo`'s GPU
training path on Apple Silicon (tested on an M4 Max, Mojo 1.0.0b3). It is the
companion document to [`ai_assisted_optimizations_and_benchmarks.md`](ai_assisted_optimizations_and_benchmarks.md),
which documents the NVIDIA/CUDA optimization journey that reached parity with
`llm.c`. This document covers what happens when you take that CUDA-optimized
codebase and run it on Metal — the surprises, the root causes, and the fixes.

**Related documentation:** see the NVIDIA campaign doc for the baseline kernel
designs, the optimization log, and the parity story. This document picks up
where that campaign created vendor branches.

**Validation gate:** throughout this document "test passes" means `make test`
(`test_gpt2 gpu` path) runs green. That is the ground-truth correctness gate
for all GPU changes. The equivalence suite (`make test`) exercises deliberately
odd shapes that the training loop never sees, making it the right canary for
invariant-based fast paths.

**Probe suite:** `/Users/evanowen/Workspace/scripts/llmm-metal-probes/` contains
standalone `pixi run mojo run` scripts that isolate each gotcha below. Run any of
them from the repo root.

---

## Why CUDA and Metal differ (the background model)

Before cataloging what broke, it helps to have a mental model of three fundamental
differences between the CUDA and Metal programming models that explain most of the
surprises.

**Address-space model.** CUDA has a *unified generic address space*: a pointer
without an explicit address-space qualifier can legally point into device-global,
shared (SMEM), local, or constant memory; the hardware figures out which at
runtime. Metal AIR (Apple Intermediate Representation) has *separate named address
spaces*: `device` (global GPU memory), `threadgroup` (shared memory), `constant`,
and `thread` (register). A pointer annotated `device` cannot legally dereference
`threadgroup` data and vice versa. If you cast a `threadgroup` pointer to `device`
(the Metal AIR equivalent of GENERIC) the compiler does *not* error; it silently
generates a load from device memory, producing zeros or garbage because the address
isn't in the device address space at all.

**Queue model.** A CUDA `DeviceContext` exposes multiple streams that execute
concurrently; cross-stream dependencies require explicit events or fences. Metal's
`DeviceContext` maps to a *single in-order command queue*: every kernel launch and
`enqueue_copy` submitted to the same context executes in submission order. There
is therefore no GPU-to-GPU ordering hazard between kernels on the same Metal
queue; `ctx.synchronize()` is only needed before the CPU reads back results or
before reusing a staging buffer from the host side.

**Host pointer visibility.** CUDA's unified virtual address (UVA) and cuMemAllocHost
register host memory pages so the GPU can DMA-read them directly. Metal has a
similar facility (`MTLBuffer` with shared storage mode), but an *unregistered*
host pointer — one allocated with plain `malloc` or Mojo's `alloc()` — is
invisible to the Metal command pipeline. Passing such a pointer into a kernel
produces no error; the GPU reads zeros. `DeviceContext.enqueue_create_host_buffer`
creates a Metal-registered staging buffer; `DeviceContext.enqueue_create_buffer`
creates a GPU-resident `DeviceBuffer`. Only pointers into one of these managed
objects are valid GPU arguments.

---

## Gotcha catalog

### Silent-wrong-results class (the dangerous ones)

These are the worst class of bug: the code compiles, runs to completion, and
returns plausible-looking numbers — just wrong ones.

---

#### G1 — HostBuffer pointers passed into Metal kernels read zeros

**What breaks:** The GPU encoder, backward bucket passes, and the loss-readback path
all need token indices (`inputs`, `targets`) and bucket metadata
(`bucket_info`, `workload_indices`) in GPU kernels. On CUDA these are loaded from
`HostBuffer` pointers: the driver registers the host memory with `cuMemAllocHost`
and the GPU can DMA-read it. On Metal, a `HostBuffer`'s raw pointer is not
registered as a Metal resource; the kernel silently reads zeros from it.

**Effect:** The encoder would look up token-embedding row 0 for every position
(embedding the null token everywhere). The loss classifier would score against
target 0 everywhere. The WTE backward would accumulate all gradients into row 0 of
`dwte`. None of this causes a crash or NaN — the model just learns nothing.

**Fix:** Allocate parallel `DeviceBuffer`s for each of these arrays, upload via
`enqueue_copy` before each forward, and pass the device-resident pointers to GPU
kernels. The `HostBuffer` originals remain for CPU-side reads.

**Code sites:**
- `train_gpt2.mojo:655–671` — field declarations; the comment at line 657 explains
  the split: `inputs_buf`/`targets_buf` (host, CPU-only) alongside
  `inputs_dev_buf`/`targets_dev_buf` (device, Metal reads).
- `train_gpt2.mojo:1526–1575` — the `enqueue_copy` calls that upload inputs and
  bucket metadata before each forward; Metal path uses device buffers, CUDA path
  uses the host pointers directly (the `comptime if not HAS_METAL` guards).

**Probe:** `probe_encoder.mojo` in the probe suite validates that the encoder
forward produces the correct output when given device-resident token indices.

---

#### G2 — `linalg.matmul` `elementwise_lambda_fn` epilogue produces scrambled output on Metal

**What breaks:** Mojo's `linalg.matmul` accepts an `elementwise_lambda_fn` that
runs a user-supplied function on every output element after the GEMM, allowing
bias addition and GELU to be "fused" into the matmul call. On NVIDIA this epilogue
is lowered to a *separate elementwise kernel* (not truly fused into the GEMM; see
entry ⑫ in the NVIDIA campaign doc) but the output is correct. On Metal, the
epilogue runs but produces *scrambled output* — the bias values are applied to wrong
positions, the GELU is computed on garbage, and the result looks numerically
reasonable but is wrong element-by-element.

The root cause has not been isolated to a specific AIR lowering bug, but the probe
reproduces it reliably: any `matmul_fwd` call with `has_bias=True` diverges from
the CPU reference on Metal when the epilogue lambda is used, while `has_bias=False`
(plain GEMM, no lambda) is correct.

**Fix:** On Metal (`not HAS_CUBLAS` and `HAS_METAL`), drop the epilogue lambda
entirely and follow the plain GEMM with a standalone `bias_gelu_fwd` kernel that
reads the GEMM output and adds the bias (and optionally GELU) in a separate pass.
This is the same split that `USE_GELU_FUSION=False` uses on CUDA, just made the
*only* path on Metal.

This fix is the `eed66d1` restructure pattern: trainer call sites in
`train_gpt2.mojo` remain single un-branched calls to `matmul_fwd`; the vendor
split lives *inside* `llmm/matmul.mojo::matmul_fwd` at the dispatch point
(`comptime if HAS_CUBLAS ... else ... comptime if HAS_METAL`), keeping the
training loop vendor-agnostic.

**Code site:** `llmm/matmul.mojo:682–705` — the Metal branch calls plain
`matmul[transpose_b=True, target=target]` then `bias_gelu_fwd`, with no host
fences (stream-ordered, see G13).

**Probe:** `probe_matmul_bias.mojo` — runs `matmul_fwd` with `has_bias=True` and
`has_bias=False` on GPU and CPU at the real QKV projection shape (B=4, T=64,
C=768, OC=2304), compares the outputs, and reports max absolute error. On Metal
the epilogue path fails with large error; the split path passes.

---

#### G3 — `address_space_cast[GENERIC]()` on SHARED pointers silently corrupts threadgroup loads/stores on Metal AIR

**What breaks:** Throughout the original kernel implementations, SHARED
(`AddressSpace.SHARED`) `LayoutTensor` pointers were cast to the GENERIC address
space via `address_space_cast[AddressSpace.GENERIC]()` before being rebounded to a
plain `MutKernelPtr`. On CUDA this is harmless — CUDA's GENERIC space includes
shared memory, so the cast is a no-op and the loads/stores land correctly in SMEM.
On Metal AIR, GENERIC means *device* (global GPU memory). The Metal compiler
accepts the cast without error, but every load from the resulting pointer reads
from device memory at that address — which holds whatever happens to be there
(usually zeros or stale data) — and every store goes to device memory instead of
the intended threadgroup location.

**Effect:** In `attention.mojo`'s flash-forward kernel, the shared Q/K/V tiles
were loaded and the scores computed using SHARED pointers cast to GENERIC. The
loaded values were all zero, producing all-zero dot products, then softmax of all
zeros (uniform 1/T), then all-uniform attention weights. The output was not NaN
(uniform attention is a valid-looking but wrong result), making this extremely hard
to spot from loss curves alone. In `encoder.mojo`'s `wte_backward_gpu_kernel`,
the same pattern applied to the accumulator shared-memory buffer, causing gradient
accumulation to write to device memory instead of SMEM and the subsequent reduction
to read zeros — resulting in zero WTE gradients.

**Fix:** Use `LayoutTensor.ptr` directly, or rebind *preserving the
`AddressSpace.SHARED` annotation*:

```mojo
# Wrong (corrupts Metal): strips SHARED, redirects to device memory
var smem_ptr = rebind[MutKernelPtr[dtype]](
    smem_tensor.ptr.address_space_cast[AddressSpace.GENERIC]()
)

# Correct: preserves SHARED annotation, threadgroup loads/stores work
var smem_ptr = rebind[
    UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED]
](smem_tensor.ptr)
```

Or access the tensor's `.ptr` field directly without any cast.

**Code sites:**
- `llmm/attention.mojo:510–513` — the removal comment explaining why the former
  `_attention_shared_memory_row_pointer` helper was deleted; the Metal fix comment
  at line 531–532 explains the rebind-to-SHARED pattern used in all replacement
  sites.
- `llmm/attention.mojo:829–831` — the in-kernel fix comment showing the pattern.
- `llmm/encoder.mojo:629–635` — the same fix in `wte_backward_gpu_kernel`, with a
  comment citing both fix points (early-return and address-space cast).

**Probe:** `probe_smem_direct.mojo` — runs two kernels side-by-side at the same
SMEM shape: Pattern A uses `smem_tensor.ptr` directly (correct on both targets),
Pattern B uses `address_space_cast[GENERIC]` (broken on Metal). Expected output for
Pattern A is 10.0 (sum of first 4 values); Pattern B reads zeros and returns 0.0
on Metal.

---

#### G4 — Early `return` before `barrier()` leaves undefined shared memory state on Metal

**What breaks:** Mojo/Metal's `barrier()` lowers to `threadgroup_barrier(mem_threadgroup)`,
a *collective* operation: the Metal memory model requires **all threads in the
threadgroup** to reach it. If any thread exits the kernel via an early `return`
before the barrier, the behavior is undefined — on Metal hardware, threads that
do reach the barrier may see stale or garbage shared-memory values because the
fence never completed.

In `wte_backward_gpu_kernel`, the original code had:
```
if c >= channels: return         # guard for out-of-bounds channel group
if warp_id >= bucket_size: return  # guard for inactive warps
...
barrier()                         # UNSAFE: not all threads reach this
```

On CUDA this worked coincidentally: hardware warp scheduling ensures warps that
never issued a `bar.sync` don't block warps that do (CUDA's `__syncthreads` is
not truly a collective in the per-thread sense for early-exit paths). On Metal,
the `threadgroup_barrier` is a strict collective and the early exits caused
incorrect results — the barrier-protected SMEM reduction produced wrong gradients.

**Fix:** Replace early returns with boolean flags that gate the per-thread work
while ensuring all threads still reach the barrier:

```mojo
var c_valid = c < channels
var active = warp_id < bucket_size

# All threads write a zero into their SMEM slot regardless of c_valid/active
for k in range(width):
    accum_shared.ptr[tid * width + k] = 0.0

if c_valid and active:
    # ... accumulate ...
    for k in range(width):
        accum_shared.ptr[tid * width + k] = accum[k]

barrier()   # Now ALL threads reach this — safe on Metal

if c_valid and warp_id == 0:
    # ... reduce ...
```

Inactive threads do a trivial extra zero-write before reaching the barrier. On
NVIDIA the computed results are bit-identical; only the control-flow structure
changes.

**Code site:** `llmm/encoder.mojo:610–619` — the Metal fix comment; lines 620–692
show the flag-based pattern with `barrier()` at line 692.

Note: this is not Metal-specific in principle. The same pattern is latent undefined
behavior on any GPU where threadgroup barriers are strict collectives. The NVIDIA
path happened to work due to CUDA's more permissive `bar.sync` semantics.

---

#### G5 — `syncwarp()` provides NO memory fence on Metal

**What breaks:** Mojo's `syncwarp()` intrinsic is intended to synchronize threads
within a warp/SIMD group. On CUDA it lowers to `__syncwarp()`, which provides both
execution convergence *and* a memory fence for warp-shared data. On Apple Silicon,
`syncwarp()` lowers to `llvm.air.simdgroup.barrier(Int32(0), Int32(4))`, which is
`simdgroup_barrier(mem_flags::mem_none)` — an *execution-only* barrier with
**no memory ordering guarantee** (`mem_none = 0`).

The effect is that writes to shared memory from one SIMD lane are not guaranteed
to be visible to other lanes after a `syncwarp()` on Metal. The reason it sometimes
works is that Apple M-series hardware executes SIMD groups in strict lock-step —
stores complete before the execution barrier fires, so the reads happen to see the
correct values. But this is a hardware implementation detail, not a Metal spec
guarantee, and it could change on future hardware.

**Correct pattern:** Always use `barrier()` (which lowers to
`threadgroup_barrier(mem_threadgroup)`) for any handoff of threadgroup-memory data
between SIMD lanes. `syncwarp()` on Metal is safe only for pure execution
convergence where no shared memory is read after the sync.

**Probe:** `probe_syncwarp.mojo` — runs two kernels: one using `syncwarp()` for a
thread-0-writes, all-read SMEM handoff (coincidentally correct on M-series due to
lock-step, but annotated as UNSAFE), and one using `barrier()` (guaranteed correct).
The probe prints which behavior was observed and explains the Metal lowering.

---

### Hard-failure class

These are bugs that produce clear errors at runtime (or compile time), making them
easier to find but still worth cataloging for the "why" explanation.

---

#### G6 — Vendor BLAS (`linalg.matmul.vendor.blas`) raises at runtime on Metal

**What breaks:** The CUDA-optimized d_weight backward uses
`linalg.matmul.vendor.blas` which wraps cuBLAS/cuBLASLt (providing `transpose_a`
support that plain `linalg.matmul` lacks, issue #6626). On Metal, the same import
compiles but raises an `AsyncRT` exception at runtime: "Metal not supported" (or
similar — the exact message is from MAX's AsyncRT layer, not the Metal runtime
itself).

**Fix for d_weight:** On Metal (`HAS_METAL` path in `matmul_d_weight_bwd`), use
pure `linalg.matmul` with an explicit transpose of the *smaller* operand as a
pre-step, then fold the result with a tiled transpose-add kernel.

The key insight is operand selection: to compute
`d_weight[OC, C] = d_outputᵀ[OC, rows] @ input[rows, C]` without `transpose_a`,
you can either:
- Transpose `d_output` (size `rows × OC`, e.g. 4096 × 50304 ≈ 206 M elements
  for the lm_head layer), or
- Transpose `input` (size `rows × C`, e.g. 4096 × 768 ≈ 3.1 M elements)

When `C < OC` (the lm_head case), transposing `input` is ~66× less work.
The result is `d_weight_Tᵀ[C, OC]` which is then folded into `d_weight[OC, C]`
via `_gpu_transpose_add_into_kernel` (see G15 for the bank-conflict detail).
When `C >= OC` (the linear projection case: C=3072, OC=768), the original
transpose-d_output strategy is used.

**Code site:** `llmm/matmul.mojo:1447–1600` — the HAS_METAL branch of
`matmul_d_weight_bwd` with the C-vs-OC decision at line 1466 and the comment
explaining the element counts.

---

#### G7 — MAX `comm/allreduce` (ZeRO collectives) fail to instantiate on Apple

**What breaks:** `ZeroContext.allreduce` / `reducescatter` / `allgather` use
MAX's `comm.allreduce` library, which in `mojo 1.0.0b3` has only implemented the
Lamport-based allreduce protocol for NVIDIA hardware (it requires CUDA P2P, signal
buffers allocated with CUDA primitives, and so on).

On Apple Silicon (or any non-NVIDIA GPU), calling `allreduce` in multi-GPU mode
raises: `"Multi-GPU collectives require Nvidia GPUs; not supported on this hardware"`.
For single-GPU training (the only meaningful configuration on Apple Silicon anyway)
all of these code paths are gated on `N >= 2` and never execute.

**Fix:** The `comptime if has_nvidia_gpu_accelerator()` guards already in
`llmm/zero.mojo:265,337,405,570` ensure the NVIDIA-specific paths never compile
on Apple. Single-GPU training is fully functional. The `ZeroContext.__init__`
method also skips signal-buffer initialization on non-NVIDIA hardware
(`llmm/zero.mojo:221`).

**Code site:** `llmm/zero.mojo:221,265,302,337,404,570,632` — all multi-GPU
collective paths are guarded by `comptime if has_nvidia_gpu_accelerator()`.

---

#### G8 — `comptime for` over 4×32 iterations in flash-attention softmax crashes the Metal AIR backend

**What breaks:** The flash-attention forward kernel's inner KV loop was originally
written as a `comptime for key_column in range(Bc)` inside an outer
`comptime for local_row in range(ROWS_PER_WARP)`. With `Bc=32` and
`ROWS_PER_WARP=4`, this fully unrolls to 128 iterations inline in the function
body. The Metal LLVM/AIR backend has a function-body size limit; exceeding it
produces an internal compiler error ("Metal Compiler failed to compile metallib")
at `ctx.compile_function` time.

**Fix:** Make the KV loop a *runtime* loop. Since `USE_GEMM_ATTENTION` is `True`
on Metal (HAS_METAL is set), the flash-forward kernel is only reached on the
portable non-GEMM path, which is never called on NVIDIA. Removing the `comptime`
unroll has zero effect on NVIDIA forward performance.

**Code site:** `llmm/attention.mojo:811–821` — the comment at line 811 reads:
"NOTE: This is a runtime loop (not comptime) on purpose. The Metal GPU compiler
(MetalAIRPass) crashes when the inner KV loop is fully unrolled at compile-time
inside the already-unrolled outer `comptime for local_row` loop."

---

#### G9 — 32 KB threadgroup memory limit enforced at compile time on Metal

**What breaks:** The flash-attention backward kernels were tuned with NVIDIA tile
sizes (Br=16, Bc=16 for bf16) that require shared-memory allocations larger than
32768 bytes per threadgroup. Metal enforces a hard 32 KB threadgroup memory limit
at PSO-creation time (the Metal equivalent of `ctx.compile_function`), raising a
clear error.

**Fix:** Reduce tile sizes. The backward's shared-memory budget is analyzed in the
comment at `llmm/attention.mojo:4678–4691`. For the Metal path:

```
float32 (sizeof=4): Br=8, Bc=8 → dQ=25,664 bytes, dKV=30,784 bytes ✓
bfloat16 (sizeof=2): Br=8, Bc=8 → dQ=13,376 bytes, dKV=13,376 bytes ✓
```

An earlier attempt used Br=16 for bf16 (two Metal SIMD groups per block), which
fit within 32 KB but produced intermittently corrupted gradients on M4 Max (grad
norms reaching 1e6–1e14 in training). The single-simdgroup Br=8 geometry is fully
validated. bf16 therefore uses the same Br=8/Bc=8 as fp32 (see G19 for the
two-simdgroup story).

**Code site:** `llmm/attention.mojo:4672,4692–4693` — the hard limit comment and
the `comptime Br = 8`, `comptime Bc = 8` values.

---

#### G10 — Raw `alloc()` pointers rejected as `enqueue_copy` destinations on Metal

**What breaks:** `test_gpt2.mojo` originally used plain `alloc[dtype](n)` to
create comparison buffers, then called `ctx.enqueue_copy` to copy GPU results into
them. On Metal, `enqueue_copy` requires that the destination be a Metal-registered
buffer (a `HostBuffer` or `DeviceBuffer` managed by the context). A raw
heap-allocated pointer is not registered with the Metal command pipeline and the
call raises: "Invalid Metal buffer pointer" (or equivalent from the Metal
validation layer).

**Fix:** Allocate the copy destination with
`ctx.enqueue_create_host_buffer[dtype](n)` so it is Metal-registered:

```mojo
# test_gpt2.mojo:37-39
# Allocate a Metal-registered host buffer so enqueue_copy works on
# both Metal (which rejects plain-malloc dst pointers) and CUDA.
var host_buf = ctx.enqueue_create_host_buffer[dtype](n)
```

**Code site:** `test_gpt2.mojo:37–43` — the comment and fix; the original
`alloc`-based pattern is explained at line 220–226 for the logits comparison path
(same fix applied there).

---

### Performance findings

These are not bugs but architectural discoveries that required rethinking kernel
design choices that were correct on NVIDIA but wrong on Metal.

---

#### P11 — Flash-attention kernels ran at <1% of FLOP peak on Metal

**Finding:** The pure-Mojo flash-attention kernels that drove the early NVIDIA
optimization campaign (`_attention_fwd_flash`, `_attention_bwd_*`) ran at
essentially zero useful throughput on Metal. The root cause: every attention score
`S_ij` is computed as a scalar dot product over `head_dim=64` elements in a loop,
with a `warp.sum` call to reduce the partial sums. This pattern uses zero
matrix-unit (shader core) resources — it is pure scalar arithmetic, and Apple
Silicon's GPU has no instruction-level parallelism on scalar paths.

GEMM-decomposed attention (batching all B·NH heads into a single `linalg.matmul`
call per phase — QKᵀ, softmax, A·V) was 8–10× faster because `linalg.matmul`
dispatches to Metal's matrix-multiplication shaders, which are the efficient
compute path on Apple GPU.

PyTorch MPS makes the same architectural choice: its `torch.nn.functional.scaled_dot_product_attention` uses MPS's batched GEMM primitives rather than flash-style scalar loops, which is why the PyTorch MPS reference is a stronger baseline than pure-Mojo flash on this hardware.

**Fix:** `USE_GEMM_ATTENTION = HAS_CUBLAS or HAS_METAL` in `llmm/attention.mojo:1232`.
When Metal is detected, the GEMM-decomposed `attention_fwd_gemm` /
`attention_bwd_gemm` paths are used. The flash path is kept for non-Metal,
non-NVIDIA environments where BLAS is unavailable.

**Code site:** `llmm/attention.mojo:1232` — the `comptime` constant that gates
the GEMM path.

---

#### P12 — Store-P (KV-cache for softmax probs): recompute path found numerically wrong

**Original finding (July 2026):** On NVIDIA, storing the `[B·NH, T, T]` softmax
probability matrix `P` per layer (the `att_probs` activation) and reading it in the
backward avoids recomputing the QKᵀ GEMM — saving ~7 ms per step. On Metal, the
original plan was to disable this store: one layer's `P` in fp32 at B=4, NH=12,
T=1024 requires 4 × 12 × 1024 × 1024 × 4 bytes ≈ **201 MB per layer**, and on
M4 Max unified memory the read-back cost was expected to exceed the QKᵀ recompute
cost.

**Correctness-campaign update (July 2026):** During the 25-step training validation
(see the training-correctness campaign section below), the Metal QKᵀ-recompute
attention backward path was found to be **numerically incorrect**. The recompute
backward amplifies gradients per layer by approximately 1.4×, compounding
geometrically with depth. With 12 layers, this produces block-grad errors up to
~460× the reference, and 25-step training that first drops then rises (the classic
dead-gradient + exploding signature). The store-P path does not have this problem.

**Resolution (July 2026, follow-up):** The recompute *math* was never wrong. The
real bug was **scratch-buffer aliasing**: a Metal fast-path in
`attention_bwd_gemm` (`llmm/attention.mojo`, since removed) assumed the shared
`gemm_att` scratch still held *this layer's* forward `P` — but forward and
backward run as two separate whole-model loops, so at backward time the scratch
holds only the **last** layer's `P`. Every layer except the last computed
`dS = scale·P·(dP − D)` with the wrong layer's probabilities, producing the
~1.4×/layer amplification. With the aliasing path removed, the true per-layer
QKᵀ recompute (`_attention_bmm_scoreout` + `pds_recompute` from the saved
log-sum-exp) passes 16/16 gradient tensors under the hardened test.

**Current state:** Store-P remains enabled — a rigorous same-tree A/B showed
recompute is a net **~3.5% loss** (fp32 736.6→762.9 ms, bf16 587.1→606.8 ms):
with the optimized GEMM kernels, forward writes `P` once either way, so skipping
the store saves nothing while recompute adds a per-layer QKᵀ GEMM in backward.
The (now-correct) recompute path is one flip away (`att_probs_addr` gates at
`train_gpt2.mojo:~1760/~2352`) if the T²-scaling store (~2.4 GB fp32 at T=1024)
ever matters more than ~3.5% step time. Lesson: "recompute vs store" tradeoffs
are not portable across kernel generations — re-measure after every major kernel
change.

---

#### P13 — Metal's single in-order queue: GPU-to-GPU ordering fences are unnecessary

**Finding:** The NVIDIA campaign's portable GPU path kept `ctx.synchronize()`
fences between kernel phases in `matmul_fwd` and `attention_fwd_gemm` because
CUDA streams are concurrent: without fences, a downstream kernel can start before
its producer finishes, causing `CUDA_ERROR_ILLEGAL_ADDRESS`. On Metal, the
`DeviceContext` is a single in-order command queue: every `enqueue_function` and
`enqueue_copy` executes in submission order. There is therefore no GPU-to-GPU race
between kernels on the same Metal context.

Removing the inter-kernel fences on the Metal path saved approximately 90 ms/step
in the trainer (the synchronize calls were blocking the CPU pipeline) plus
additional time in `matmul_fwd`. The fences are preserved on the CUDA portable
path where they remain load-bearing.

**Re-validation note (correctness campaign, July 2026):** All 8 Metal-gated sync
removals were confirmed non-load-bearing during the 25-step correctness campaign.
The `test_gpt2` suite (16/16 gradient tensors) passes with the sync removals in
place; the training bugs found during the campaign were unrelated to these fences
(they were backward-pass gradient-flow errors, not ordering hazards). See the
training-correctness campaign section for details.

The `CUDA_ERROR_ILLEGAL_ADDRESS` comment in `llmm/matmul.mojo:603–605` documents
why the fence exists for CUDA: "empirically required on the GPU target — without
them this raced against neighboring kernels." The `comptime if not HAS_METAL`
guards at lines 690–694 strip those fences on the Metal path.

**Code sites:**
- `llmm/matmul.mojo:688–694` — the Metal fast path skips both `ctx.synchronize()`
  calls with `comptime if not HAS_METAL`.
- `llmm/matmul.mojo:1463–1464` — same pattern in `matmul_d_weight_bwd`: "All ops
  are stream-ordered on Metal; no host fence needed."
- `train_gpt2.mojo:1764,1888,1926,2038,2319` — various forward/backward fence
  guards, all `comptime if not HAS_METAL`.

---

#### P14 — Transpose the smaller operand for d_weight on Metal

**Finding:** Computing `d_weight[OC, C] = d_outputᵀ @ input` without `transpose_a`
support requires pre-transposing one operand. The naive choice — transpose
`d_output` (the output gradient, shape `[rows, OC]`) — is reasonable when
`rows × OC ≈ rows × C`, but in the lm_head layer `OC = 50304` (vocabulary) while
`C = 768`: transposing `d_output` means moving `4096 × 50304 ≈ 206 M elements`,
while transposing `input` moves only `4096 × 768 ≈ 3.1 M elements`. The transpose
cost dominates on a bandwidth-constrained GPU; choosing the smaller operand reduces
transpose traffic by a factor of ~66×.

The result `d_weight_T[C, OC]` is then folded into `d_weight[OC, C]` via a tiled
shared-memory transpose kernel (G15).

**Code site:** `llmm/matmul.mojo:1449–1464` — the decision comment, the
element-count analysis, and the `if in_channels < out_channels` branch.

---

#### P15 — Tiled 32×32 shared-memory transpose with 32×33 padding eliminates bank conflicts

**Finding:** The `_gpu_transpose_add_into_kernel` in `matmul.mojo` folds a
`d_weight_T[C, OC]` scratch into `d_weight[OC, C]` while transposing. A naive
out-of-place transpose produces non-coalesced stores (threads in a warp scatter
to non-adjacent addresses). A shared-memory-staged tiled transpose reads coalesced
from global memory into SMEM, then writes coalesced from SMEM to global memory.

The standard 32×32 tile has a bank-conflict hazard: 32 threads in a column read
SMEM slots that are 32-element-stride apart, landing on the same bank (for 32-bank
SMEM with 32-wide warps). The fix is well-known: pad the shared-memory tile to
32×33. The extra column ensures adjacent threads in the transposed dimension read
from different banks, eliminating the conflict.

Result: a non-coalesced scatter that ran at ~45 ms was replaced with a coalesced
tiled transpose running at ~4 ms on the same shape.

**Code site:** `llmm/matmul.mojo:1304–1385` — `_gpu_transpose_add_into_kernel`; the
`comptime STRIDE = TILE + 1` (i.e., 33) padding at line 1330, the bank-conflict
explanation in the docstring at line 1317. Note also the Metal-specific safety
comment at line 1323: "Metal fix: tile.ptr[i] is used directly to preserve
AddressSpace.SHARED; GENERIC address-space casts corrupt threadgroup pointers on
Metal AIR" (G3 applied here too).

---

#### P16 — M4 Max thermal throttling: benchmark discipline on Apple Silicon

**Finding:** Sustained GPU load on an M4 Max causes significant thermal throttling
after approximately 8 seconds of heavy computation. During benchmarking sessions,
step time for PyTorch MPS was observed to increase from ~877 ms to 1500–2500 ms
within a few steps of starting a sustained workload. The throttling is
significantly more pronounced than on the GB10 DGX Spark (which throttles after
much longer durations due to active cooling).

**Discipline for Metal benchmarks:**
- Use short runs (3–5 steps maximum) from a cold GPU state.
- Always measure the first 1–2 steps before throttling engages.
- Compare only measurements taken within the same thermal window (same run).
- Report the GPU temperature and clock state alongside measurements.
- Do not compare a "first step of run A" against a "step 5 of run B".

Use `make benchmark-metal BENCH_METAL_STEPS=5` and take the median of steps 1–3
(step 0 includes JIT compilation overhead on some paths).

---

#### P17 — MFU on Apple Silicon: no tensor cores; bf16 at fp32 ALU rate

**Finding:** Apple Silicon GPU cores are SIMD ALUs with no dedicated matrix
multiplication accelerator analogous to NVIDIA's Tensor Cores. All numeric
precisions — fp32, fp16, bf16 — share the same ALU throughput. This has two
implications:

1. `bf16` costs the same compute as `fp32` per operation. The only advantage of
   bf16 is bandwidth: half as many bytes per element means twice as many elements
   per memory transfer. For memory-bound kernels this can help; for compute-bound
   kernels it is neutral (unlike NVIDIA where bf16 on Tensor Cores is 2× faster
   than TF32 on the same units).

2. The MFU (Model FLOPs Utilization) denominator is the device's peak *fp32*
   throughput, and the same value is used for both fp32 and bf16 builds.

The `llmm/mfu.mojo` MFU table uses substring matching on the Metal device name
(`llmm/mfu.mojo:265–303`, Pass 2) to look up Apple Silicon entries, with peak
values derived from independent benchmark databases (notebookcheck, nanoreview,
cpu-monkey) using the formula:
```
peak_fp32 = gpu_cores × 128 ALUs/core × 2 FLOP/cycle × clock_GHz
```
For M4 Max (40 GPU cores, ~1.58 GHz sustained): `40 × 128 × 2 × 1.58e9 ≈ 16.2 TFLOPS`.

**Code site:** `llmm/mfu.mojo:117–170` — the Apple-specific comments, the `_apple`
helper function (identity scaling, tf32 == bf16_32), and the database entries for
M1–M5 family.

---

#### P18 — DeviceAttribute queries work through the Metal HAL

**Finding:** The NVIDIA campaign's occupancy heuristics used
`ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)` to size the kernel grid
to the number of SMs. This query was expected to be NVIDIA-specific, but it works
on Apple Silicon too — Mojo's `DeviceContext` HAL exposes this attribute through
the Metal backend, where `MULTIPROCESSOR_COUNT` returns the number of GPU cores
(40 on an M4 Max).

The same is true for `MAX_THREADS_PER_BLOCK` (1024 on Metal, matching the Metal
specification for maximum threads per threadgroup) and
`MAX_SHARED_MEMORY_PER_BLOCK` (32768 bytes = 32 KB, matching G9's observed
limit).

The backward kernel's grid sizing (`num_blocks = max(min(num_tiles, SM_OVERPROVISION × num_sm), 1)`)
therefore ports unchanged to Metal.

**Code sites:** `llmm/attention.mojo:4701–4702` — `get_attribute(MULTIPROCESSOR_COUNT)`
in the backward kernel grid calculation; also used at lines 1547, 1640, 3865, 3901, 4390.

---

#### P19 — Two-simdgroup blocks diverge on Metal with the bf16 flash backward

**Finding (documented dead end):** During the Metal port, the bf16 flash-backward
kernel was tested with `Br=16` (two Metal simdgroups per threadgroup, 64 threads),
which fits within the 32 KB SMEM limit and gives better occupancy in theory. In
practice, with two simdgroups sharing the same threadgroup and communicating via
SMEM, the bf16 backward produced intermittently corrupted gradients on M4 Max:
gradient norms ranging from 1e6 to 1e14 (vs expected ~1–10) during training.

The fp32 single-simdgroup (Br=8, 32 threads) configuration was fully validated and
produced correct results. The hypothesis is that the two-simdgroup bf16 path
exposed a subtle interaction between bf16 rounding, the accumulator reduction
pattern, and Metal's SIMD-group barrier semantics — but the exact root cause was
not isolated. The fix was to use Br=8 for bf16 as well, matching the validated
fp32 geometry.

This is an example of a kernel tile geometry that validates on one target (NVIDIA)
but fails on another (Metal) in a way that isn't caught by kernel unit tests — it
only manifests under sustained training where gradient accumulation amplifies the
small numerical error. The training-loss poisoning took ~2 steps to become apparent.

**Code site:** `llmm/attention.mojo:4687–4692` — the comment "bf16 previously used
Br=16 (BLOCK_SIZE_DQ=64, two Metal simdgroups); that configuration produced
intermittently corrupted dQ/dK/dV on Apple M4 Max."

---

#### P20 — Headout-GEMM 4-rows-per-thread tile chosen over 2-rows variant

**Finding:** A 2-rows-per-thread headout kernel (`_attn_headout2_gpu`, BM=32,
BK=32, BN=64) was benchmarked alongside the 4-rows-per-thread
`_attn_headout4_gpu` (BM=64, BK=16, BN=64). The 2-row variant has half the
barrier count (32 vs 64 outer iterations) but requires two passes to load the
B-tile (BK×BN=2048 = 2×THREADS). In practice the 4-row layout showed equal or
better throughput because the larger BM=64 fills exactly THREADS=1024 for the
A-tile load (100% efficiency, no padding), and BK=16 shrinks the inner loop and
reduces register pressure.

**Outcome:** `_attn_headout2_gpu` and its launcher `_launch_headout2` were
removed from `attention.mojo` as dead code (neither was ever called from the
active dispatch path in `_attention_bmm_headout`).

---

---

## The training-correctness campaign

> **Action required — re-validate on NVIDIA GB10:** All three training bugs
> described below (C1–C3) were target-independent: the same incorrect gradient
> flow ran on CUDA, the portable GPU path, and Metal alike. Prior GB10
> training-quality conclusions (loss trajectories, convergence plots) should be
> considered provisional until re-validated with the hardened `test_gpt2`
> (`make test` on GB10 after updating from this commit). Performance conclusions
> (GB10 step time, llm.c parity) are unaffected — those were measured from a
> cold model before any gradient accumulation.

### What the original test missed

`test_gpt2` was the declared correctness gate throughout the Metal port campaign,
but on close inspection it had two gaps that made it unable to catch training-
dynamics bugs:

1. **Loose absolute-only gradient tolerances.** Every gradient tensor was checked
   with `maxdiff <= 2.0` — an absolute floor with no relative component. For
   parameters like `wpe` whose reference magnitude is O(1e-4), a computed value of
   0.0 has `maxdiff ≈ 1e-4`, well below the threshold. The test passed while the
   gradient was six orders of magnitude too small.

2. **10-step loss trajectory computed but discarded.** The loop ran 10 training
   steps and computed `model.mean_loss` at each step, but the only assertion was on
   step 0's logits and loss. Steps 1–9 were run for weight-update side-effects only;
   whether the loss was falling or exploding was never checked.

A 25-step training run on fresh (non-debug-state) data exposed rising validation
loss from step ~10 onward — the classic signature of a broken gradient chain.
Bisecting the backward pass revealed three distinct bugs.

### C1 — Residual-skip gradient never seeded before layernorm_fused_residual_bwd

**What was wrong:** `layernorm_fused_residual_bwd` is a fused kernel that computes
the LayerNorm input gradient and accumulates it into the two residual-sum inputs
(`d_inp1 += LN_dinp; d_inp2 += LN_dinp`). The forward operation is
`out = LayerNorm(inp1 + inp2)`, so the *true* input gradient is
`LN_dinp + d(inp1+inp2)`, where `d(inp1+inp2)` is the incoming residual-stream
gradient flowing back from the layer above. The fused kernel only performs the
`+= LN_dinp` part. The `d(inp1+inp2)` term — the residual identity skip — was
never added before the fused kernel ran, so every layer's backward consumed
gradients that were missing the carry term from above.

**Effect:** Block gradients decayed geometrically toward layer 0. In the worst
case, `dwpe` — which depends on all layers' skip carries — was approximately
10⁶× smaller than the reference. The test missed this because `|dwpe_computed|
≈ 1e-4 ≈ |dwpe_reference|` by coincidence (both near the absolute tolerance floor).

**Fix:** A new `residual_grad_broadcast` kernel (`llmm/layernorm.mojo:2122`) seeds
the incoming residual gradient into both targets before the fused backward runs:
`d_inp1 += src; d_inp2 += src`. It is called at three points in
`train_gpt2.mojo::backward()`:
- `train_gpt2.mojo:2304` — before the LN2 fused backward (all layers): seeds
  `d_block_input` and `d_l_attn_proj` from `d_l_fc_proj`.
- `train_gpt2.mojo:2424` — layer 0, after LN1 backward: seeds the LN1 skip into
  `d_block_input` / `d_l_attn_proj` from the LN1 input-gradient scratch.
- `train_gpt2.mojo:2440` — layers > 0, before the inter-layer propagation:
  seeds `residual_2[L-1]` / `fc_proj[L-1]` from `d_block_input`.

**This bug predated the Metal port and affected all targets (CUDA, GB10, CPU).**

### C2 — GELU gradient fused into the wrong matmul backward

**What was wrong:** GPT-2's MLP block has two matmuls: FC (`C → 4C`, produces
`fch`) and PROJ (`4C → C`, consumes `fch_gelu = gelu(fch)`). The GELU
nonlinearity sits between them at the 4C boundary. Its gradient must be fused into
the backward of whichever matmul *crosses* that boundary — which is the PROJ
backward (going from `d_fc_proj` to `d(fch_gelu)` to `d(fch)`). The FC backward
lives entirely below the GELU (it takes `d_fch` and propagates to `d_ln_2`) and
must use `use_gelu=False`.

The bug was that `use_gelu=True` was passed to the FC matmul backward and
`use_gelu=False` to the PROJ backward — exactly backwards. The effect: the FC
backward applied `gelu'` to `fch_gelu` (a 4C-wide tensor) using a C-wide stride,
corrupting `d_ln_2`. The `check_tensor` call at step 0 caught a maxdiff of 15.76
on `dln2w` once the L2-ratio guard was added.

**Fix:** Swap `use_gelu` between the two MLP matmul backward calls
(`train_gpt2.mojo:2257` — PROJ backward, now `use_gelu=True`; `train_gpt2.mojo:2277`
— FC backward, now `use_gelu=False`). The comment at line 2247–2255 explains the
dataflow.

**This bug predated the Metal port and affected all targets (CUDA, GB10, CPU).**

### C3 — Layer-0 LN1 backward passed the normed output instead of the pre-norm input

**What was wrong:** `layernorm_bwd` reconstructs `xhat = (input - mean) / rstd`
to compute the gamma gradient: `dgamma[c] = Σ xhat[i,c] · d_out[i,c]`. This
requires the **pre-norm input** (the raw activations before LayerNorm). For all
layers > 0, the correct pre-norm input (`residual_2[L-1]`, the sum
`block_input + attn_proj`) was passed. For layer 0, the code accidentally passed
`l_ln_1` — the normed **output** — instead of `acts.encoded` (the encoder output,
which is the pre-norm input for layer 0's LN1).

**Effect:** `dgamma` (the gamma/scale gradient for LN1) was wrong at layer 0;
`dbeta` (which only depends on `d_out`, not `xhat`) was correct. The error was
small enough that a flat absolute tolerance of 2.0 passed it.

**Fix:** Pass `self.acts.encoded` as the LayerNorm input at the layer-0 LN1
backward site (`train_gpt2.mojo:2408–2410`). The comment at lines 2403–2406
explains the requirement.

**This bug predated the Metal port and affected all targets (CUDA, GB10, CPU).**

### Metal-only find: recompute-QKᵀ attention backward path numerically wrong

During the campaign, the Metal recompute backward path (which avoids storing the
`[B·NH, T, T]` softmax matrix `P` by re-running QKᵀ in the backward) was found
to amplify gradients by approximately 1.4× per layer. With 12 layers the
compounding error reaches ~1.4¹² ≈ 12× at the bottom layer and, after passing
through wpe, the gradient norm is orders of magnitude off. The 25-step run showed
a loss that initially fell correctly then rose sharply — consistent with an
exploding-gradient signature after the optimizer consumed the corrupted step.

The store-P path (writing `att_probs` in the forward and reading it in the
backward) does not have this problem.

**Resolved:** the follow-up investigation found the recompute math itself was
correct — the failure was a scratch-buffer aliasing fast-path that fed every
layer's backward the *last* layer's `P` (see the P12 entry above for the full
story). The aliasing path is removed, the recompute backward now passes 16/16
under the hardened test, and store-P remains the default because a same-tree
A/B measured recompute as a ~3.5% net step-time loss.

### Test hardening

The following changes were made to `test_gpt2.mojo` to close the gaps above:

1. **Mixed atol + rtol tolerance** (`test_gpt2.mojo:89, 204–206`): each gradient
   tensor now uses `maxdiff <= GRAD_ATOL + GRAD_RTOL * ref_maxabs` with
   `GRAD_ATOL = 0.01`, `GRAD_RTOL = 0.05`. For a typical tensor with
   `ref_maxabs ≈ 2`, the threshold is 0.11 — tight enough to catch corruptions
   that the old flat 2.0 threshold passed.

2. **L2 norm ratio guard** (`test_gpt2.mojo:95–98`): `our_l2` must be within
   `GRAD_L2_FACTOR = 3.0` of `ref_l2` in both directions. A dead-gradient whose
   L2 is 10⁶× too small fails immediately even if its element-wise maxdiff is
   below the threshold.

3. **Loss-trajectory assertion** (`test_gpt2.mojo:591–595`): the 10-step loss
   sequence is now checked at every step with `LOSS_STEP_TOL = 0.01`, matching
   llm.c's criterion. A dead-gradient or exploding-gradient failure causes the
   loss to stall or diverge, failing the check at step 1.

4. **Fixture regeneration** (`test_gpt2.mojo:295–306`): the `expected_losses`
   array was regenerated by
   `~/Workspace/scripts/llmm-metal-probes/gen_expected_losses.py`
   (cited in the in-file comment at `test_gpt2.mojo:290`). The old fixture used
   mismatched optimizer hyperparameters (β₂ 0.95 vs 0.999, weight_decay 0 vs
   0.01), so the expected trajectory did not match the test loop's actual update
   rule.

### Lessons

- **Single-step gradient checks with absolute tolerances are not enough.** A
  parameter whose gradient is 10⁶× too small can still have `maxdiff ≈ 0` if the
  reference magnitude is also small. Use relative tolerances and L2 ratio guards.
- **Overfit-single-batch tests mask data-pipeline and step-coupling issues.** The
  debug-state batch runs the same data each step, hiding bugs that manifest only
  under gradient accumulation or when residual carries compound over steps.
- **Assert the loss trajectory.** Gradient bugs that leave step-0 correct but
  break step-1 onward are caught only if you check all steps.
- **L2 norm ratio guards catch dead gradients instantly.** A tensor that is
  geometrically decaying to zero will fail the L2 ratio guard even if its maxdiff
  is small.

---

## Performance timeline (Metal, B=4, T=1024, M4 Max)

Intermediate milestones measured with the profile harness / short training runs
(median of post-warmup steps, cold GPU); the final rows are the official
`make benchmark-metal` result after the correctness campaign.

| milestone | step (ms) | tok/s | key change |
|-----------|----------:|------:|------------|
| Initial Metal port (flash attention, host buffers fixed — G1–G5) | ~3627 | ~1125 | First green `test_gpt2 gpu`; flash-attention path, no GEMM |
| GEMM-decomposed attention (P11) | ~1290 | ~3162 | `USE_GEMM_ATTENTION=True` on Metal; 8–10× attention speedup |
| Sync audit: remove GPU-to-GPU fences (P13) | ~1200 | ~3413 | `comptime if not HAS_METAL` fence removal |
| d_weight transpose algebra + fence removal | ~857 | ~4782 | P14 combined |
| Tiled transpose-add fold + attention micro-tuning | ~816 | ~5019 | P15; attention bwd fence removal |
| Pre-correctness-campaign fp32 snapshot | ~739 | ~5555 | 2026-07-02 23:14; numbers survived gradient fixes |
| Post-correctness benchmark (2026-07-03 01:51) | 737.74 fp32 / 587.81 bf16 | 5552 / 6968 | First all-green official run; store-P re-enabled |
| Vectorized bias_gelu (P21) | −40 both precisions | — | 2D-grid SIMD kernel, 30→320-770 GB/s; helps every non-cuBLAS forward matmul |
| Attention-bwd transA GEMM (P22) | −51 bf16 / −57 fp32 | — | Folded operand-transpose into register-tiled GEMM; killed 4 transposes + generic kvgrad |
| **Official benchmark: llm.mojo fp32** | **652.06** | **6282** | `make benchmark-metal`, 2026-07-03 10:48, cold GPU, 30s inter-arm cooldowns |
| **Official benchmark: llm.mojo bf16** | **498.92** | **8210** | Same run — **fastest config; 7.3× the initial Metal port** |
| **Official benchmark: PyTorch MPS fp32** | **830.27** | **4933** | Same run, same thermal window — llm.mojo fp32 is 27% faster |
| **Official benchmark: PyTorch MPS bf16** | **857.26** | **4778** | Same run — **llm.mojo bf16 is 1.72× faster than PyTorch MPS bf16** |

Figure: `figures/benchmark_metal_b4_t1024_2026-07-03_1048_Apple-M4-Max_Mac-M4-Max.png`
(An earlier 04:39 run measured 643/496/816/846 — the ±1.5% between runs is
normal thermal variance; the 10:48 run is canonical because its figure carries
the fixed chart layout.)

Remaining honest headroom (from the 2026-07-03 comprehensive re-profile): the
fused-residual layernorm (~78 ms/step) is structurally barrier/occupancy-bound —
five optimization attempts measured slower or numerically unsafe and were
reverted; adamw (478–510 GB/s), global_norm, fused_classifier, and the core
GEMMs (at the linalg library ceiling) are at their floors. The fp32 LM-head
d_input GEMM (~82 ms at 24% peak, shape-limited) remains the largest fp32-only
target; bf16 halves it. Further significant gains likely require custom GEMM
kernels beyond `linalg.matmul` or Modular-side improvements.

> **Note:** The intermediate rows are development measurements; the official rows
> come from `scripts/benchmark_train.py` via `make benchmark-metal` (cold GPU,
> all four arms measured with 30-second inter-arm cooldowns in the same session —
> see P16). bf16 is the fastest config: the earlier observation that bf16 was
> *slower* than fp32 (~1030+ ms) was an artifact of the broken Metal
> QKᵀ-recompute attention backward path. With store-P re-enabled (the recompute
> path disabled), bf16 benefits from half the memory bandwidth pressure and is
> 20% faster than fp32 on Metal.

---

## Per-operation profiling methodology on Metal

Unlike NVIDIA (where `ncu` provides kernel-level hardware counters and `nsys`
provides a timeline), Apple Silicon has different tooling:

| tool | available | use |
|------|-----------|-----|
| `make profile-metal` (Perfetto in-process) | YES | Full step timeline, per-kernel spans |
| `make benchmark-metal` | YES | Throughput (ms/step) vs PyTorch MPS |
| NVIDIA Nsight Compute (`make profile-ncu`) | NO | Exits cleanly with a note |
| NVIDIA Nsight Systems (`make profile-nsys`) | NO | Exits cleanly with a note |
| Metal GPU Frame Capture (Xcode) | Manual | Deep per-dispatch profiling |

**Sync-window batching for micro-benchmarks:** When timing individual kernels on
Metal, the same methodology as the NVIDIA campaign applies — batch N iterations
inside one sync window and divide by N — but with stricter thermal discipline due
to M4 Max's faster throttling (P16):

```mojo
# Warmup passes (3) — let JIT and caches settle
for _ in range(3):
    ctx.enqueue_function(kernel, ...)
    ctx.synchronize()

# Measurement (10 launches, one sync window)
var t0 = global_perf_counter_ns()
for _ in range(10):
    ctx.enqueue_function(kernel, ...)
ctx.synchronize()
var t1 = global_perf_counter_ns()
print("mean:", (t1 - t0) / 10 / 1e6, "ms")
```

See `probe_time_attn.mojo`, `probe_time_matmul.mojo`, and `probe_time_tadd.mojo`
in the probe suite for the canonical pattern.

**Thermal discipline:** Before any comparative measurement: let the GPU idle for
30 seconds, verify the GPU fan is not spinning (on MacBook) or that the device
temperature is back at idle (~30–40°C as reported by a system monitor). The M4 Max
does not expose GPU temperature directly in software; use a third-party tool (e.g.
`asitop` or Instruments' GPU occupancy view) to confirm the thermal state.

---

## How to validate

**Ground-truth gate:** `make test` runs `test_gpt2 gpu` which checks activations
and gradients against the PyTorch reference. Always run this after any Metal kernel
change. The equivalence suite (`tests/`) exercises odd shapes that the training
loop never uses, so it catches invariant-based fast paths that accidentally break
non-standard configurations (see G3, G4 effects on the suite).

```sh
# Full test suite (includes GPU correctness)
make test

# GPU forward/backward equivalence only
make verify     # fp32; validates all 16 gradient tensors

# Profile harness (confirm step time without 40-step thermal exposure)
make build-profile
LLMM_PROFILE_B=4 LLMM_PROFILE_T=1024 LLMM_PROFILE_LAYERS=12 LLMM_PROFILE_STEPS=4 \
  pixi run ./build/profile_gpt2_bf16 gpu

# Throughput vs PyTorch MPS
make benchmark-metal BENCH_B=4 BENCH_T=1024 BENCH_METAL_STEPS=5

# Force the portable GPU path on NVIDIA hardware (exercises the Metal code path)
LLMM_FORCE_PORTABLE_GPU=1 make verify
```

**Running individual probes:**

```sh
# From the repo root, any probe in the suite:
pixi run mojo run -I . /Users/evanowen/Workspace/scripts/llmm-metal-probes/probe_smem_direct.mojo
pixi run mojo run -I . /Users/evanowen/Workspace/scripts/llmm-metal-probes/probe_matmul_bias.mojo
pixi run mojo run -I . /Users/evanowen/Workspace/scripts/llmm-metal-probes/probe_syncwarp.mojo
pixi run mojo run -I . /Users/evanowen/Workspace/scripts/llmm-metal-probes/probe_encoder.mojo
```

---

## NVIDIA side notes: what stays CUDA-only and why

This section is included for readers moving between the two backends.

**cuBLASLt fused epilogues (`HAS_CUBLAS` path in `matmul_fwd`):** The NVIDIA path
fuses the bias add and optional GELU directly into the cuBLASLt GEMM via epilogue
codes `GELU_AUX_BIAS` (164), `BIAS` (4), and `DGELU` (192). This is a genuine
GEMM-internal fusion that saves a full 206 M-element elementwise pass per forward.
It is NVIDIA-specific: the cuBLASLt epilogue API has no Metal equivalent. The Metal
path does the same work in two steps: plain GEMM + `bias_gelu_fwd` kernel. This
costs one extra memory pass but is correct and approximately equivalent in terms of
step time because the Metal path is memory-bandwidth-limited on both passes.

**TensorCore MMA path in attention:** The CUDA attention uses
`cublasGemmStridedBatchedEx` (or cuBLASLt strided-batched, same underlying kernel)
to batch all B·NH heads into a single GEMM call on CUDA's Tensor Cores. Metal uses
`linalg.matmul` batched with a loop over B·NH heads; each head call dispatches to
Metal's matrix shaders. The per-head structure is similar in spirit but the actual
tensor-core utilization numbers differ — Apple Silicon has no "tensor%" metric
comparable to NVIDIA ncu's.

**Store-P KV-cache on CUDA:** Entry ㉔ in the NVIDIA campaign doc documents that
storing softmax probabilities per layer is a net win on the GB10 (saves ~7 ms of
QKᵀ recompute at the cost of 1.2 GB of activation memory). The original plan was
to disable this on Metal to avoid the 201 MB/layer bandwidth cost (P12 above).
However, the correctness campaign found the Metal recompute backward path to be
numerically incorrect (~1.4× per-layer gradient amplification), so store-P is now
**unconditionally enabled** for all targets (`train_gpt2.mojo:1769, 2352`). The
re-enabled store-P also makes bf16 the fastest Metal config (see the updated
performance timeline). The recompute backward fix is a tracked follow-up item.

**GPU-to-GPU ordering fences:** The fences at `matmul_fwd` and `matmul_d_input_bwd`
(CUDA portable path) guard a real race: without them, `CUDA_ERROR_ILLEGAL_ADDRESS`
is reproducible. On Metal, the in-order queue makes those fences unnecessary and
removing them saves CPU-stall time. The guards (`comptime if not HAS_METAL`) keep
the CUDA behavior intact.

**`HAS_CUBLAS` / `HAS_METAL` / `LLMM_FORCE_PORTABLE_GPU` flag semantics:**
`llmm/vendor.mojo` is the single source of truth. `HAS_CUBLAS` is True only on
NVIDIA hardware without `LLMM_FORCE_PORTABLE_GPU=1`. `HAS_METAL` is True on Apple
Silicon without `LLMM_DISABLE_METAL=1`. Both can be False simultaneously (e.g.
AMD GPU), in which case the portable `linalg.matmul` path runs without Metal
optimizations. `LLMM_FORCE_PORTABLE_GPU=1` on NVIDIA forces the Metal/portable
path to run on CUDA hardware — useful for testing the Metal code path in CI where
no Apple hardware is available.

**The `eed66d1` pattern:** Vendor-specific workarounds live inside the op files
(`matmul.mojo`, `attention.mojo`) behind `comptime if HAS_CUBLAS / HAS_METAL`
guards. Call sites in `train_gpt2.mojo` remain single un-branched calls to
`matmul_fwd`, `attention_fwd`, etc. This keeps the training loop vendor-agnostic:
adding a new backend means editing the op files, not the trainer.

---

## AI use statement

The Apple Silicon port and optimization campaign documented in this file (July 2026)
were performed with AI assistance via Claude Code, under the direction of Evan Owen.
The probe suite (`/Users/evanowen/Workspace/scripts/llmm-metal-probes/`) was
developed to isolate and reproduce each gotcha before and after the fix.

The NVIDIA optimization campaign that preceded this port is documented in
[`ai_assisted_optimizations_and_benchmarks.md`](ai_assisted_optimizations_and_benchmarks.md).
