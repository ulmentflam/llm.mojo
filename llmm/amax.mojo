# ===----------------------------------------------------------------------=== #
# amax.mojo — GPU amax reduction + delayed per-tensor scaling state.
#
# Chunk C of docs/ai/fp8_training_design.md (§1.3 "Scaling strategy", §3 "the
# dtype-generic low-precision layer"). The design places this machinery
# inside llmm/lowp.mojo alongside `PrecisionSpec`/`quantize_devscale`
# (Chunks A/B/G, developed in parallel worktrees touching that same file).
# DELIBERATE DEVIATION: to keep parallel agents from editing the same file,
# this chunk lives in its own `llmm/amax.mojo` for now. Folding this file's
# contents into `llmm/lowp.mojo` (or vice versa) at merge time is a
# mechanical move, not a redesign — every symbol here is written exactly as
# the design's §3 prototypes describe (`AmaxState[spec]`, `compute_amax[spec,
# in_dtype]`, `update_scale`), just under a different module path.
#
# GPU-only, unconditionally. Everything in this file assumes it is only ever
# called from an already `LOWP_ENABLED and is_gpu[target]()`-gated call site
# (design §5); unlike llmm/global_norm.mojo (which serves both the fp32-CPU
# and bf16-GPU training paths and so carries its own `is_cpu`/`is_gpu`
# comptime dispatch), fp8/fp4 training is GPU-only from the flag axis down
# (design §2, landmine #1: AArch64 codegen crashes if any "cpu" target is
# instantiated under low precision), so nothing here needs — or should have —
# a CPU code path. Do not add one; the gate is the caller's, not this file's.
#
# Determinism note (why `compute_amax`'s reduction needs no fixed topology to
# be provably deterministic, unlike llmm/global_norm.mojo's float *sum*):
# floating-point `max` (and NaN-infectious `max`, as implemented below) is
# *exactly* associative and commutative in IEEE 754 — max(max(a,b),c) ==
# max(a,max(b,c)) bit-for-bit, with no rounding step to make the result
# order-dependent. Float *addition* (global_norm's reduction) is not exactly
# associative under rounding, so global_norm.mojo has to commit to one fixed
# grid/block partitioning and argue determinism from "same topology every
# run". `compute_amax` picks a fixed topology too (same BLOCK_SIZE/grid-size
# formula every call for a given tensor size and SM count), but the stronger
# claim holds regardless: any reduction topology over the same multiset of
# per-element (abs-value, is-bad) pairs yields the identical amax, because
# `max` and the "was any element non-finite" OR-reduction (itself implemented
# as a max over a {0.0, 1.0} indicator, exact for the same reason) are both
# order-independent operators. Determinism is inherent to the operator here,
# not contingent on the chosen launch topology.
#
# NaN/Inf handling (not spelled out verbatim in the design; this is Chunk C's
# concrete interpretation, called out explicitly for chunk D/E/F reviewers):
# `compute_amax` treats any non-finite input element as "infectious" — the
# per-element abs-value is excluded from the max (contributes the identity
# 0.0, never corrupts the running max via a NaN comparison, which in IEEE 754
# is always false and would otherwise silently make max() ignore the NaN) —
# but a separate "saw a non-finite element" flag is reduced alongside it with
# the same order-independent max-of-{0,1} trick, and applied once at the very
# end: if any element was non-finite, `amax_out[0]` is written as NaN. This
# makes a corrupted tensor unmistakable at the amax level (`isnan(amax)`
# rather than a silently-wrong finite number) instead of either crashing or
# quietly discarding the bad elements. `update_scale` then treats a NaN/Inf
# (or non-positive, e.g. all-zero-tensor) amax uniformly: it falls back to a
# finite `scale = 1.0` rather than propagating Inf/NaN into the scale that
# every future quantize call multiplies by (`0/0` and `x/0` are exactly the
# div-by-zero / NaN-scale failure modes the design's own Gate C rules out).
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import ceildiv, isnan, isinf, nan
from std.sys import simd_width_of
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives import block

from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.lowp import PrecisionSpec, ScalingKind


# ===----------------------------------------------------------------------=== #
# Small pointer-mutability bridge.
# ===----------------------------------------------------------------------=== #


@always_inline
def kernel_ptr_as_immut[
    dtype: DType
](ptr: MutKernelPtr[dtype]) -> ImmutKernelPtr[dtype]:
    """`MutKernelPtr -> ImmutKernelPtr` view, for feeding a `compute_amax`
    output scratch buffer (necessarily `Mut`, since `compute_amax` writes it)
    into `AmaxState.update_scale` (which only reads `amax_current`). Mirrors
    `llmm/memory.mojo`'s `as_immut_kernel_from_mut`, which does the same
    conversion for the `MutUntrackedOrigin`-flavored `MutMemPtr` instead of
    this file's `MutAnyOrigin`-flavored `MutKernelPtr`.
    """
    return rebind[ImmutKernelPtr[dtype]](ptr.as_immutable())


@always_inline
def device_buf_mut_ptr[
    dtype: DType
](buf: DeviceBuffer[dtype]) -> MutKernelPtr[dtype]:
    """`DeviceBuffer.unsafe_ptr()` returns a pointer tied to the buffer's own
    (tracked) origin; every kernel-launch call site in this file wants the
    type-erased `MutAnyOrigin` flavor (`MutKernelPtr`) that raw GPU kernel
    args use throughout llmm/. Widening that implicitly is a soft-deprecated
    compiler warning (`Implicitly converting an UnsafePointer to
    MutUnsafeAnyOrigin`), so make the widen explicit here — mirrors the
    `.as_unsafe_any_origin()` + `rebind` pattern already used in
    tests/test_zero.mojo for the same conversion.
    """
    return rebind[MutKernelPtr[dtype]](buf.unsafe_ptr().as_unsafe_any_origin())


# ===----------------------------------------------------------------------=== #
# format_max — max finite representable magnitude per low-precision format.
# ===----------------------------------------------------------------------=== #


def format_max[dtype: DType]() -> Float32:
    """Max finite representable magnitude for a low-precision GEMM operand
    format — the delayed-scaling target in `scale = fmt_max / (amax /
    2^margin)` (design §1.3). Comptime-parameterized by `dtype` (not baked
    into `PrecisionSpec`) so one `AmaxState[spec]` site can be told whether it
    is quantizing to `spec.fwd_dtype` (E4M3) or `spec.bwd_dtype` (E5M2)
    without two separate specs.

    `float4_e2m1fn` (NVFP4's element format, max magnitude 6.0) is included
    now even though FP4 quantization itself is an unbuilt seam (§3 extension
    seam #1) — it is a single constant, not machinery, so there is nothing to
    stub.
    """

    comptime if dtype == DType.float8_e4m3fn:
        return 448.0
    elif dtype == DType.float8_e5m2:
        return 57344.0
    elif dtype == DType.float4_e2m1fn:
        return 6.0
    else:
        comptime assert (
            False
        ), "format_max: unsupported/unrecognized low-precision dtype"


# ===----------------------------------------------------------------------=== #
# compute_amax — GPU reduction: bf16/fp32 tensor -> fp32 amax (PerTensor).
# ===----------------------------------------------------------------------=== #


@always_inline
def _amax_partial_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int,
](
    partial_max_ptr: MutKernelPtr[DType.float32],
    partial_bad_ptr: MutKernelPtr[DType.float32],
    data_ptr: ImmutKernelPtr[dtype],
    count: Int,
) -> None:
    # Deliberately unaligned SIMD loads (no `alignment=` hint) — this is a
    # reduction over a possibly ragged/odd-stride slice (a GEMM operand
    # transient, not a fixed-shape buffer this file controls), and asserting
    # an alignment the compiler can't prove is exactly the
    # CUDA_ERROR_MISALIGNED_ADDRESS landmine documented in
    # docs/ai/gpt2-bf16-generation-misaligned-address (memory). Correctness
    # over peak bandwidth here — this kernel runs once per site per step on
    # activation-sized tensors, not in the GEMM inner loop.
    var index = (block_idx.x * block_dim.x + thread_idx.x) * width
    var grid_width = block_dim.x * grid_dim.x * width

    var local_max = SIMD[DType.float32, width](0.0)
    var local_bad = SIMD[DType.float32, width](0.0)
    var local_max_tail = Scalar[DType.float32](0.0)
    var local_bad_tail = Scalar[DType.float32](0.0)

    var idx = index
    while idx < count:
        if idx + width <= count:
            var val = (data_ptr + idx).load[width=width]().cast[DType.float32]()
            var is_bad = isnan(val) | isinf(val)
            var abs_val = abs(val)
            # NaN-infectious max (see module docstring): zero out bad lanes
            # before folding into local_max (a NaN compared against anything
            # is always False, so an un-guarded max() would silently drop
            # it); track "saw a bad lane" separately via a max-of-indicator,
            # which is itself exact/order-independent for the same reason
            # plain max is.
            local_max = max(
                local_max,
                is_bad.select(SIMD[DType.float32, width](0.0), abs_val),
            )
            local_bad = max(
                local_bad,
                is_bad.select(
                    SIMD[DType.float32, width](1.0),
                    SIMD[DType.float32, width](0.0),
                ),
            )
        else:
            for j in range(idx, count):
                var v = data_ptr[j].cast[DType.float32]()
                if isnan(v) or isinf(v):
                    local_bad_tail = 1.0
                else:
                    local_max_tail = max(local_max_tail, abs(v))
        idx += grid_width

    var thread_max = max(local_max.reduce_max(), local_max_tail)
    var thread_bad = max(local_bad.reduce_max(), local_bad_tail)

    var block_max_val = block.max[block_size=BLOCK_SIZE](thread_max)
    var block_bad_val = block.max[block_size=BLOCK_SIZE](thread_bad)

    if Int(thread_idx.x) == 0:
        var out_index = Int(block_idx.x)
        partial_max_ptr[out_index] = block_max_val
        partial_bad_ptr[out_index] = block_bad_val


def _amax_aggregate_gpu[
    BLOCK_SIZE: Int,
](
    amax_out_ptr: MutKernelPtr[DType.float32],
    partial_max_ptr: MutKernelPtr[DType.float32],
    partial_bad_ptr: MutKernelPtr[DType.float32],
    grid_size: Int,
) -> None:
    var tid = Int(thread_idx.x)
    var local_max = Scalar[DType.float32](0.0)
    var local_bad = Scalar[DType.float32](0.0)
    var idx = tid
    while idx < grid_size:
        local_max = max(local_max, partial_max_ptr[idx])
        local_bad = max(local_bad, partial_bad_ptr[idx])
        idx += BLOCK_SIZE

    var total_max = block.max[block_size=BLOCK_SIZE](local_max)
    var total_bad = block.max[block_size=BLOCK_SIZE](local_bad)
    if tid == 0:
        amax_out_ptr[0] = nan[DType.float32]() if total_bad > 0.5 else total_max


def compute_amax[
    spec: PrecisionSpec,
    in_dtype: DType,
](
    amax_out: MutKernelPtr[DType.float32],
    data: ImmutKernelPtr[in_dtype],
    size: Int,
    ctx: DeviceContext,
) raises -> None:
    """Device reduction: `amax_out[0] = max(|data[0:size]|)`, fp32, over a
    bf16 or fp32 `in_dtype` source tensor (design §3's `compute_amax[spec,
    in_dtype](x, amax_out, ctx)`). Non-finite inputs make the result NaN (see
    module docstring); an all-zero (or empty) tensor gives a legitimate 0.0,
    which `update_scale` — not this function — guards against turning into a
    div-by-zero scale.

    PerTensor (fp8) only; `spec.scaling == Block1D/Block2D` (NVFP4's
    per-block reduction, §3 seam) is not implemented — comptime error rather
    than a silently-scalar result for a block-scaled spec.

    DELIBERATELY STAYS UNIMPLEMENTED for FP4 (reconciled during the FP4-GEMM
    chunk, docs/ai/fp4_training_recipes_research.md): `llmm/nvfp4_quant.mojo`'s
    `nvfp4_quantize` does not call `compute_amax`/`AmaxState` at all. It
    computes its own fp32 per-tensor scale fresh every call, fully
    device-resident, via its own two-pass reduction
    (`nvfp4_compute_tensor_scale`) — and its e4m3 per-block scales are
    derived in the same quantize kernel launch from that fresh tensor scale,
    with no separate reduction pass. There is no delayed/history-based
    scaling step for FP4 to plug into this file's machinery: fp8's
    `PerTensor` case needs `AmaxState`'s ring buffer because a single
    per-tensor scale is coarse and benefits from smoothing across
    `amax_history_len` steps; NVFP4's 16-element block scale already adapts
    *within* a tensor, so amortizing the coarser tensor-level scale over
    steps buys little and was not built. Filling in `Block1D`/`Block2D` here
    would be parallel, unused machinery — see `FP4_SPEC` in `llmm/lowp.mojo`
    for the corresponding note on the spec-constant side of this decision.
    """

    comptime if spec.scaling == ScalingKind.PerTensor:
        comptime BLOCK_SIZE = 512
        comptime RESIDENT_THREADS = 2048
        comptime width = simd_width_of[in_dtype]()

        var num_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        var max_grid = (num_sm * RESIDENT_THREADS) // BLOCK_SIZE
        var needed_grid = ceildiv(size, BLOCK_SIZE * width)
        var grid_size = max(1, min(max_grid, needed_grid))

        var partial_max = ctx.enqueue_create_buffer[DType.float32](grid_size)
        var partial_bad = ctx.enqueue_create_buffer[DType.float32](grid_size)

        comptime partial_kernel = _amax_partial_gpu[in_dtype, BLOCK_SIZE, width]
        var compiled_partial = ctx.compile_function[partial_kernel]()
        ctx.enqueue_function(
            compiled_partial,
            device_buf_mut_ptr(partial_max),
            device_buf_mut_ptr(partial_bad),
            data,
            size,
            grid_dim=(grid_size,),
            block_dim=(BLOCK_SIZE,),
        )

        comptime aggregate_kernel = _amax_aggregate_gpu[BLOCK_SIZE]
        var compiled_aggregate = ctx.compile_function[aggregate_kernel]()
        ctx.enqueue_function(
            compiled_aggregate,
            amax_out,
            device_buf_mut_ptr(partial_max),
            device_buf_mut_ptr(partial_bad),
            grid_size,
            grid_dim=(1,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        comptime assert False, (
            "compute_amax: Block1D/Block2D (FP4) per-block reduction is an"
            " unimplemented seam (docs/ai/fp8_training_design.md §3) —"
            " PerTensor (fp8) only in Chunk C"
        )


# ===----------------------------------------------------------------------=== #
# AmaxState — per-GEMM-operand-site delayed-scaling state (device-resident).
# ===----------------------------------------------------------------------=== #


def _amax_state_init_gpu[
    history_len: Int,
](
    history_ptr: MutKernelPtr[DType.float32],
    scale_ptr: MutKernelPtr[DType.float32],
    scale_inv_ptr: MutKernelPtr[DType.float32],
    static_scale: Float32,
) -> None:
    var tid = Int(thread_idx.x)
    if tid < history_len:
        history_ptr[tid] = 0.0
    if tid == 0:
        # A1 (docs/ai/speedrun_techniques_research.md): `static_scale > 0`
        # seeds scale/scale_inv from a calibrated constant (llmm/lowp.mojo's
        # `fp8_static_scale`) instead of the placeholder 1.0/1.0 — this is
        # the ONLY write `AmaxState.scale`/`scale_inv` ever receive under
        # `-D LLMM_FP8_STATIC_SCALES=1` (matmul.mojo's `matmul_fwd_lowp`/
        # `matmul_bwd_lowp` skip `update_scale`/`update_scale_pair` entirely
        # in that mode — see the module comment above `FP8_STATIC_SCALES`).
        # `static_scale <= 0` (including the default -1.0 sentinel) is the
        # ordinary dynamic path: placeholder scale/scale_inv = 1.0/1.0,
        # never meant to be consumed for a real quantize before the first
        # `update_scale` call — see the calling contract documented on
        # `update_scale`.
        if static_scale > Float32(0.0):
            scale_ptr[0] = static_scale
            scale_inv_ptr[0] = Float32(1.0) / static_scale
        else:
            scale_ptr[0] = 1.0
            scale_inv_ptr[0] = 1.0


def _update_scale_gpu[
    history_len: Int,
](
    history_ptr: MutKernelPtr[DType.float32],
    scale_ptr: MutKernelPtr[DType.float32],
    scale_inv_ptr: MutKernelPtr[DType.float32],
    amax_current_ptr: ImmutKernelPtr[DType.float32],
    step: Int,
    margin_mult: Float32,
    fmt_max: Float32,
) -> None:
    var amax_current = amax_current_ptr[0]

    # Compute the scale for THIS call's step from the EXISTING history —
    # i.e. strictly prior steps' amax, per design §1.3's literal wording
    # ("The scale used this step is derived from *prior* steps' amax") —
    # before this step's own amax is pushed into that same history below.
    # Getting this ordering backwards (push-then-max) would let a step's own
    # amax leak into its own scale post-warmup, which is not delayed scaling
    # (see tests/test_amax.mojo's steady-state/ring-buffer cases, which
    # pinned this down after an initial reversed-order draft failed them).
    var amax_for_scale: Float32
    if step < history_len:
        # Warmup (design §1.3: "use current scaling ... compute this
        # tensor's amax just-in-time") — this step's own amax, not the
        # (still partially-empty) history.
        amax_for_scale = amax_current
    else:
        # Steady state: delayed scaling from the full (pre-push) history
        # window (`amax_compute_algo = max`, §1.3). NaN/Inf-infectious for
        # the same reason as compute_amax (module docstring).
        var m = Float32(0.0)
        var bad = False
        for i in range(history_len):
            var h = history_ptr[i]
            if isnan(h) or isinf(h):
                bad = True
            else:
                m = max(m, h)
        amax_for_scale = nan[DType.float32]() if bad else m

    # Guarded scale formula (§1.3: `scale = fp8_max / (amax / 2^margin)`,
    # rewritten as `fmt_max * 2^margin / amax` to avoid a redundant divide).
    # `amax_for_scale <= 0` covers the legitimate all-zero-tensor case (no
    # div-by-zero); NaN/Inf is the corrupted-input detection path. Both fall
    # back to a finite `scale = 1.0` — see module docstring.
    var scale: Float32
    if (
        amax_for_scale <= Float32(0.0)
        or isnan(amax_for_scale)
        or isinf(amax_for_scale)
    ):
        scale = Float32(1.0)
    else:
        scale = (fmt_max * margin_mult) / amax_for_scale

    scale_ptr[0] = scale
    scale_inv_ptr[0] = Float32(1.0) / scale

    # Push this step's amax into the ring buffer *after* computing the
    # scale above, so future (post-warmup) calls see it but this call's own
    # scale never does — this is what makes the ring buffer hold "prior
    # steps" relative to whichever call is currently deriving a scale from
    # it.
    history_ptr[step % history_len] = amax_current


struct AmaxState[spec: PrecisionSpec](Movable):
    """Per-GEMM-operand-site delayed-scaling state (design §1.3, §3).

    PerTensor (fp8): `history` is a length-`spec.amax_history_len` fp32 ring
    buffer of past amaxes; `scale`/`scale_inv` are length-1 fp32 device
    scalars used to quantize/dequantize this site's operand. Everything lives
    on-device — no host buffer changes size, and there is no host readback in
    the normal path (design §4's landmine-2 audit: "per-site amax history +
    scale ... Host readback? no (debug only)") — so `update_scale` never
    stalls the training loop on a device->host sync. This is also *why* the
    ring buffer + scale live here rather than as plain host `Float32`s: an
    `AmaxState` per site (design: one per GEMM operand, so up to 8 per
    transformer block — 4 forward operand pairs + their backward
    counterparts — times the layer count) updated every step would mean that
    many device->host syncs per step if the state were host-resident;
    keeping it device-resident and updating it with a tiny single-thread
    kernel (`_update_scale_gpu`) costs one kernel launch instead.

    Calling contract for chunk D/E (the training-loop integrators): call
    `state.update_scale[fmt_dtype](amax_current_ptr, ctx)` once per step with
    that step's own just-computed `amax_current` (from `compute_amax` on the
    operand about to be quantized). Immediately after the call,
    `state.scale`/`state.scale_inv` are valid for **that same step's**
    quantize, in both regimes — `update_scale` derives the scale from the
    ring buffer's contents *before* pushing `amax_current` into it (design
    §1.3: "the scale used this step is derived from *prior* steps' amax"),
    so a step's own amax never leaks into its own scale:
      - **Warmup** (`state.step < spec.amax_history_len`, the first
        `amax_history_len` calls): the history is still partially empty, so
        `update_scale` falls back to "use current scaling" (§1.3) — the
        derived scale *is* this step's own `amax_current`-based value. This
        necessarily gates the GEMM on the reduction; the design explicitly
        accepts that for the short warmup window.
      - **Steady state** (`state.step >= spec.amax_history_len`): the
        derived scale is `max` over the full ring buffer as it stood before
        this call — i.e. the last `amax_history_len` *prior* steps, never
        this one. Because that computation does not read `amax_current` at
        all in this branch, a caller free to compute `amax_current`
        off the critical path (e.g. fused into the quantize/dequantize
        kernel, or from a prior step's already-quantized copy) can call
        `update_scale` without gating this step's GEMM on a fresh reduction
        — "delayed" scaling per §1.3. `amax_current` is still pushed into
        history at the end of every call (both regimes), so the ring buffer
        is always current for the next call.
      - Ring buffer indexing is `state.step % spec.amax_history_len`, so
        after `amax_history_len` steps every slot has been written exactly
        once and further pushes wrap around, overwriting oldest-first (a
        plain FIFO ring, not an "amax_compute_algo=most_recent" pointer
        chase) — see `tests/test_amax.mojo`'s ring-buffer-wrap case.

    Block1D/Block2D (NVFP4, §3 seam) is stubbed: `__init__`, `update_scale`
    raise a clear comptime error rather than silently allocating/updating the
    wrong shape. UPDATE (FP4-GEMM chunk): this stays a deliberate stub, not a
    pending TODO — `llmm/nvfp4_quant.mojo`'s `nvfp4_quantize` computes both
    scale levels itself, fresh every call, and never consults `AmaxState`.
    See `compute_amax`'s docstring above for the full rationale.

    `(Movable)` (added by Chunk D): `List[AmaxState[spec]]` — the
    per-layer/per-site storage `train_gpt2.mojo`'s `LowpState` container uses
    (one instance per GEMM operand site per transformer layer) — requires the
    element type to conform to Mojo's `Movable` trait
    (`stdlib/collections/list.mojo`'s `struct List[T: Movable]`); Mojo does
    not infer trait conformance implicitly, even when every field (here, three
    `DeviceBuffer[DType.float32]`s and an `Int`) is itself movable, so the
    trait must be declared. All fields are simple/movable and no custom
    `__moveinit__`/`__del__` exists, so the compiler-synthesized move is
    correct as-is (verified empirically: an un-annotated struct with the same
    field shapes fails `List[Foo]()`'s constraint with "has 'Movable' type,
    but value has type 'AnyStruct[Foo]'"; adding `(Movable)` alone, no
    `__moveinit__`, fixes it). No behavior change to any existing call site.
    """

    var history: DeviceBuffer[DType.float32]
    var scale: DeviceBuffer[DType.float32]
    var scale_inv: DeviceBuffer[DType.float32]
    var step: Int

    def __init__(
        out self, ctx: DeviceContext, static_scale: Float32 = Float32(-1.0)
    ) raises:
        """`static_scale` (A1, default -1.0 == dynamic/unchanged): when >0,
        seeds `scale`/`scale_inv` from this calibrated constant instead of
        the placeholder 1.0/1.0 — see `_amax_state_init_gpu`'s docstring.
        Every existing call site (all of them omit this argument) is
        byte-identical to before this parameter was added."""
        comptime if Self.spec.scaling == ScalingKind.PerTensor:
            comptime H = Self.spec.amax_history_len
            self.history = ctx.enqueue_create_buffer[DType.float32](H)
            self.scale = ctx.enqueue_create_buffer[DType.float32](1)
            self.scale_inv = ctx.enqueue_create_buffer[DType.float32](1)
            self.step = 0

            comptime init_kernel = _amax_state_init_gpu[H]
            var compiled = ctx.compile_function[init_kernel]()
            ctx.enqueue_function(
                compiled,
                device_buf_mut_ptr(self.history),
                device_buf_mut_ptr(self.scale),
                device_buf_mut_ptr(self.scale_inv),
                static_scale,
                grid_dim=(1,),
                block_dim=(max(H, 1),),
            )
        else:
            comptime assert False, (
                "AmaxState.__init__: Block1D/Block2D (FP4) scaling is an"
                " unimplemented seam (docs/ai/fp8_training_design.md §3) —"
                " PerTensor (fp8) only in Chunk C"
            )

    def update_scale[
        fmt_dtype: DType
    ](
        mut self,
        amax_current: ImmutKernelPtr[DType.float32],
        ctx: DeviceContext,
    ) raises -> None:
        """Refresh `self.scale`/`scale_inv` from the current ring-buffer
        contents (or, during warmup, from `amax_current` itself), then push
        `amax_current` (this step's freshly computed amax of the operand
        about to be quantized, from `compute_amax`) into the history ring
        buffer for future calls. `fmt_dtype` (`spec.fwd_dtype` or
        `spec.bwd_dtype`, whichever this site quantizes to) selects the
        format max via `format_max`. See the `AmaxState` calling contract
        above for warmup vs. steady-state semantics.
        """

        comptime if Self.spec.scaling == ScalingKind.PerTensor:
            comptime H = Self.spec.amax_history_len
            comptime MARGIN_MULT = Float32(1 << Self.spec.margin)
            comptime FMT_MAX = format_max[fmt_dtype]()
            comptime update_kernel = _update_scale_gpu[H]

            var compiled = ctx.compile_function[update_kernel]()
            ctx.enqueue_function(
                compiled,
                device_buf_mut_ptr(self.history),
                device_buf_mut_ptr(self.scale),
                device_buf_mut_ptr(self.scale_inv),
                amax_current,
                self.step,
                MARGIN_MULT,
                FMT_MAX,
                grid_dim=(1,),
                block_dim=(1,),
            )
            self.step += 1
        else:
            comptime assert False, (
                "AmaxState.update_scale: Block1D/Block2D (FP4) scaling is an"
                " unimplemented seam (docs/ai/fp8_training_design.md §3) —"
                " PerTensor (fp8) only in Chunk C"
            )


# ===----------------------------------------------------------------------=== #
# update_scale_pair — Optimization C (docs/ai/ai_assisted_optimizations_and_
# benchmarks.md 2026-07-10 fp8-quant-opt entry, "Optimization C" deferred
# item; landed in the 2026-07-10 opt/fp8-kernels follow-on session).
#
# `matmul_fwd_lowp` (llmm/matmul.mojo) calls `update_scale` twice back to
# back, once for `input_state` and once for `weight_state` — two SEPARATE
# 1-thread kernel launches immediately adjacent in program order, both
# already holding their own freshly-computed `amax_current` (from two prior
# `compute_amax` calls at the same call site) by the time either update runs.
# Nothing about `update_scale`'s "this step's quantize needs this step's own
# just-updated scale" ordering contract (see `AmaxState`'s calling-contract
# docstring above) requires these two SPECIFIC calls to be separate kernel
# launches — both amaxes are already on hand, and both states' resulting
# scale/scale_inv are consumed later in the SAME function
# (`lowp_gemm_devscale`), not before. `update_scale_pair` fuses them into
# ONE kernel launch (`grid_dim=(2,)`, one block per state, `block_dim=(1,)`
# — literally "a single kernel with one thread(-block) per state", scoped to
# what is actually co-resident/ready at a single call site) instead of two.
#
# Deliberately NOT a whole-step, all-144-states batch (the mission's most
# literal reading): every OTHER `AmaxState` in this training loop is updated
# alone at its own call site (`matmul_bwd_lowp`'s single `doutput_state`
# update has no sibling to pair with — dgrad/wgrad read it read-only, per
# the "once per step" contract) or would require deferring `update_scale`
# past the point where its OWN call site's quantize/GEMM already needs the
# fresh scale (a step-wide batch would need a two-pass forward/backward
# restructuring — compute every site's amax first, batch-update every
# scale, THEN quantize/GEMM everywhere — which is exactly the "real
# restructuring... expected value is low relative to the surgery required"
# the original fp8-quant-opt session already evaluated and declined for
# this same reason). Fusing the two-per-fwd-call-site pair that already
# satisfies the ordering contract for free is the simple, real, zero-risk
# subset of that idea; see the doc entry's Optimization C write-up for the
# measured launch-count delta (144 -> 96 `update_scale`-family launches/
# step) and why the wall-clock impact is expected to be negligible (the
# fused kernel is still two single-thread blocks — this was never a
# bandwidth-bound part of the family, only a launch-count one).
#
# `compute_amax`'s second-stage reduction (`_amax_aggregate_gpu`) is
# DELIBERATELY NOT folded into this batch: `compute_amax` is a tested public
# entry point (`tests/test_amax.mojo`) used standalone at every call site
# (including `matmul_bwd_lowp`'s solo `doutput` site), and folding its
# aggregate stage into a state-pair batch would mean either changing its
# return contract (breaking the existing single-state test surface) or
# adding a second, parallel two-state code path purely for this one caller
# — more surface for a family that is already ~4% of GPU time and whose
# `amax_aggregate_gpu`/`update_scale_gpu` components individually measured
# well under 1% each (Chunk F / Optimization A-B profiles). Per the
# mission's own "only if the design stays simple; don't force it" — it
# wasn't kept simple, so it stays out.
# ===----------------------------------------------------------------------=== #


def _update_scale_pair_gpu[
    history_len: Int,
](
    history0_ptr: MutKernelPtr[DType.float32],
    scale0_ptr: MutKernelPtr[DType.float32],
    scale_inv0_ptr: MutKernelPtr[DType.float32],
    amax_current0_ptr: ImmutKernelPtr[DType.float32],
    history1_ptr: MutKernelPtr[DType.float32],
    scale1_ptr: MutKernelPtr[DType.float32],
    scale_inv1_ptr: MutKernelPtr[DType.float32],
    amax_current1_ptr: ImmutKernelPtr[DType.float32],
    step: Int,
    margin_mult: Float32,
    fmt_max: Float32,
) -> None:
    # One block per state (block_dim=(1,), so each block is already a
    # single thread — no thread_idx guard needed, matching `_update_scale_
    # gpu`'s own single-thread-block style). block_idx.x selects which
    # state's pointers this block operates on; the arithmetic body below is
    # byte-for-byte `_update_scale_gpu`'s, just parameterized by the
    # selected pointer set instead of being duplicated per state.
    var is_second = Int(block_idx.x) == 1
    var history_ptr = history1_ptr if is_second else history0_ptr
    var scale_ptr = scale1_ptr if is_second else scale0_ptr
    var scale_inv_ptr = scale_inv1_ptr if is_second else scale_inv0_ptr
    var amax_current_ptr = amax_current1_ptr if is_second else amax_current0_ptr

    var amax_current = amax_current_ptr[0]

    var amax_for_scale: Float32
    if step < history_len:
        amax_for_scale = amax_current
    else:
        var m = Float32(0.0)
        var bad = False
        for i in range(history_len):
            var h = history_ptr[i]
            if isnan(h) or isinf(h):
                bad = True
            else:
                m = max(m, h)
        amax_for_scale = nan[DType.float32]() if bad else m

    var scale: Float32
    if (
        amax_for_scale <= Float32(0.0)
        or isnan(amax_for_scale)
        or isinf(amax_for_scale)
    ):
        scale = Float32(1.0)
    else:
        scale = (fmt_max * margin_mult) / amax_for_scale

    scale_ptr[0] = scale
    scale_inv_ptr[0] = Float32(1.0) / scale

    history_ptr[step % history_len] = amax_current


def update_scale_pair[
    spec: PrecisionSpec,
    fmt_dtype: DType,
](
    mut state0: AmaxState[spec],
    amax_current0: ImmutKernelPtr[DType.float32],
    mut state1: AmaxState[spec],
    amax_current1: ImmutKernelPtr[DType.float32],
    ctx: DeviceContext,
) raises -> None:
    """Fused two-state `update_scale`: identical result to calling
    `state0.update_scale[fmt_dtype](amax_current0, ctx)` followed by
    `state1.update_scale[fmt_dtype](amax_current1, ctx)` (same
    `_update_scale_gpu` arithmetic per state, same ring-buffer push order —
    each state's own history only ever sees its own amax, in the same
    step-indexed slot it would have under two separate calls), but as ONE
    kernel launch instead of two. See the module comment above
    (Optimization C) for the call-site scoping rationale and why this stays
    a 2-state fusion rather than a whole-step batch.

    Both states must be at the SAME `.step` (true for `input_state`/
    `weight_state` at a `matmul_fwd_lowp` call site — both are only ever
    advanced together, once per call, from `step=0`) — asserted defensively
    since the fused kernel takes a single `step` value for both blocks.
    `fmt_dtype` is shared too (both call sites needing this pairing quantize
    to the same format — E4M3 for fp8's forward operand pair).
    """
    debug_assert(
        state0.step == state1.step,
        (
            "update_scale_pair: state0/state1 must be at the same .step (only"
            " ever advanced together at a shared call site)"
        ),
    )

    comptime if spec.scaling == ScalingKind.PerTensor:
        comptime H = spec.amax_history_len
        comptime MARGIN_MULT = Float32(1 << spec.margin)
        comptime FMT_MAX = format_max[fmt_dtype]()
        comptime update_kernel = _update_scale_pair_gpu[H]

        var compiled = ctx.compile_function[update_kernel]()
        ctx.enqueue_function(
            compiled,
            device_buf_mut_ptr(state0.history),
            device_buf_mut_ptr(state0.scale),
            device_buf_mut_ptr(state0.scale_inv),
            amax_current0,
            device_buf_mut_ptr(state1.history),
            device_buf_mut_ptr(state1.scale),
            device_buf_mut_ptr(state1.scale_inv),
            amax_current1,
            state0.step,
            MARGIN_MULT,
            FMT_MAX,
            grid_dim=(2,),
            block_dim=(1,),
        )
        state0.step += 1
        state1.step += 1
    else:
        comptime assert False, (
            "update_scale_pair: Block1D/Block2D (FP4) scaling is an"
            " unimplemented seam (docs/ai/fp8_training_design.md §3) —"
            " PerTensor (fp8) only in Chunk C"
        )
