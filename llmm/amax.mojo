# ===----------------------------------------------------------------------=== #
# amax.mojo — GPU amax reduction + delayed per-tensor (fp8) scaling state.
#
# GPU-only, unconditionally. Callers gate on LOWP_ENABLED and
# is_gpu[target]() (AArch64 codegen crashes if any cpu target is
# instantiated under low precision). Do not add a CPU path here — the gate
# is the caller's.
#
# Unlike llmm/global_norm.mojo (which serves both the fp32-CPU and bf16-GPU
# training paths and so carries its own `is_cpu`/`is_gpu` comptime dispatch),
# fp8/fp4 training is GPU-only from the flag axis down, so nothing here
# needs — or should have — a CPU code path.
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
# NaN/Inf handling: `compute_amax` treats any non-finite input element as
# "infectious" — the per-element abs-value is excluded from the max
# (contributes the identity 0.0, never corrupts the running max via a NaN
# comparison, which in IEEE 754 is always false and would otherwise silently
# make max() ignore the NaN) — but a separate "saw a non-finite element" flag
# is reduced alongside it with the same order-independent max-of-{0,1}
# trick, and applied once at the very end: if any element was non-finite,
# `amax_out[0]` is written as NaN. This makes a corrupted tensor unmistakable
# at the amax level (`isnan(amax)` rather than a silently-wrong finite
# number) instead of either crashing or quietly discarding the bad elements.
# `update_scale` then treats a NaN/Inf (or non-positive, e.g. all-zero-tensor)
# amax uniformly: it falls back to a finite `scale = 1.0` rather than
# propagating Inf/NaN into the scale that every future quantize call
# multiplies by (`0/0` and `x/0` are exactly the div-by-zero / NaN-scale
# failure modes this rules out).
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

    `float4_e2m1fn` (max 6.0) is included though this file's machinery is
    PerTensor-only — it's a single constant.
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
    bf16 or fp32 `in_dtype` source tensor. Non-finite inputs make the result
    NaN (see module docstring); an all-zero (or empty) tensor gives a
    legitimate 0.0, which `update_scale` — not this function — guards
    against turning into a div-by-zero scale.

    PerTensor (fp8) only; `spec.scaling == Block1D/Block2D` (NVFP4's
    per-block reduction) is not implemented — comptime error rather than a
    silently-scalar result for a block-scaled spec.

    FP4 never calls compute_amax/AmaxState — nvfp4_quantize computes both
    scale levels itself, fresh every call. Block1D/Block2D per-block
    reduction is intentionally unbuilt (it would be unused machinery).
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
            "compute_amax: Block1D/Block2D (FP4) per-block reduction is"
            " unimplemented — PerTensor (fp8) only"
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
        # static_scale > 0 seeds scale/scale_inv from a calibrated constant;
        # under LLMM_FP8_STATIC_SCALES this init is the ONLY write
        # scale/scale_inv ever receive (matmul skips update_scale entirely).
        # static_scale <= 0 (default -1.0) = dynamic path: placeholder
        # 1.0/1.0, not valid until the first update_scale.
        if static_scale > Float32(0.0):
            scale_ptr[0] = static_scale
            scale_inv_ptr[0] = Float32(1.0) / static_scale
        else:
            scale_ptr[0] = 1.0
            scale_inv_ptr[0] = 1.0


@always_inline
def _scale_from_history[
    history_len: Int,
](
    history_ptr: MutKernelPtr[DType.float32],
    scale_ptr: MutKernelPtr[DType.float32],
    scale_inv_ptr: MutKernelPtr[DType.float32],
    amax_current: Float32,
    step: Int,
    margin_mult: Float32,
    fmt_max: Float32,
) -> None:
    """Single-thread scale-update arithmetic shared by `_update_scale_gpu`
    and `_update_scale_pair_gpu`: derive this step's scale from the EXISTING
    history — i.e. strictly prior steps' amax — before `amax_current` is
    pushed into that same history below. Getting this ordering backwards
    (push-then-max) would let a step's own amax leak into its own scale
    post-warmup, which is not delayed scaling (see tests/test_amax.mojo's
    steady-state/ring-buffer cases).

    Warmup (`step < history_len`): this step's own amax, not the (still
    partially-empty) history. Steady state: delayed scaling from the full
    (pre-push) history window, NaN/Inf-infectious for the same reason as
    `compute_amax` (module docstring).

    Guarded scale formula (`scale = fmt_max / (amax / 2^margin)`, rewritten
    as `fmt_max * 2^margin / amax` to avoid a redundant divide).
    `amax_for_scale <= 0` covers the legitimate all-zero-tensor case (no
    div-by-zero); NaN/Inf is the corrupted-input detection path. Both fall
    back to a finite `scale = 1.0` — see module docstring.
    """
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

    # Push this step's amax into the ring buffer *after* computing the
    # scale above, so future (post-warmup) calls see it but this call's own
    # scale never does — this is what makes the ring buffer hold "prior
    # steps" relative to whichever call is currently deriving a scale from
    # it.
    history_ptr[step % history_len] = amax_current


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
    _scale_from_history[history_len](
        history_ptr,
        scale_ptr,
        scale_inv_ptr,
        amax_current,
        step,
        margin_mult,
        fmt_max,
    )


struct AmaxState[spec: PrecisionSpec](Movable):
    """Per-GEMM-operand-site delayed-scaling state.

    PerTensor (fp8): `history` is a length-`spec.amax_history_len` fp32 ring
    buffer of past amaxes; `scale`/`scale_inv` are length-1 fp32 device
    scalars used to quantize/dequantize this site's operand. Everything lives
    on-device — no host buffer changes size, and there is no host readback in
    the normal path — so `update_scale` never stalls the training loop on a
    device->host sync. This is also *why* the ring buffer + scale live here
    rather than as plain host `Float32`s: an `AmaxState` per site (one per
    GEMM operand, so up to 8 per transformer block — 4 forward operand pairs
    + their backward counterparts — times the layer count) updated every
    step would mean that many device->host syncs per step if the state were
    host-resident; keeping it device-resident and updating it with a tiny
    single-thread kernel (`_update_scale_gpu`) costs one kernel launch
    instead.

    Calling contract: call
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

    Block1D/Block2D (NVFP4) raises a comptime error — FP4 computes its
    scales in nvfp4_quantize and never uses AmaxState (see `compute_amax`).

    (Movable): `List[AmaxState]` (the per-site storage) requires the
    element type to conform to Movable; Mojo does not infer it. Fields are
    all movable and there is no custom `__moveinit__`/`__del__`, so the
    synthesized move is correct (verified empirically: an un-annotated
    struct with the same field shapes fails `List[Foo]()`'s constraint with
    "has 'Movable' type, but value has type 'AnyStruct[Foo]'"; adding
    `(Movable)` alone, no `__moveinit__`, fixes it). No behavior change to
    any existing call site.
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
                "AmaxState.__init__: Block1D/Block2D (FP4) scaling is"
                " unimplemented — PerTensor (fp8) only"
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
                "AmaxState.update_scale: Block1D/Block2D (FP4) scaling is"
                " unimplemented — PerTensor (fp8) only"
            )


# ===----------------------------------------------------------------------=== #
# Fuses the two adjacent update_scale launches at a matmul_fwd_lowp site
# (input_state + weight_state, both already holding fresh amax, both
# consumed later in the same GEMM) into ONE launch (grid=(2,), one block per
# state). Bit-identical to two separate calls. Deliberately a 2-state
# fusion, not a whole-step batch (every other site updates alone, or a
# batch would defer a scale past where its own GEMM needs it).
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
    # state's pointers this block operates on; the arithmetic body below
    # matches `_update_scale_gpu`, just parameterized by the selected
    # pointer set instead of being duplicated per state.
    var is_second = Int(block_idx.x) == 1
    var history_ptr = history1_ptr if is_second else history0_ptr
    var scale_ptr = scale1_ptr if is_second else scale0_ptr
    var scale_inv_ptr = scale_inv1_ptr if is_second else scale_inv0_ptr
    var amax_current_ptr = amax_current1_ptr if is_second else amax_current0_ptr

    var amax_current = amax_current_ptr[0]
    _scale_from_history[history_len](
        history_ptr,
        scale_ptr,
        scale_inv_ptr,
        amax_current,
        step,
        margin_mult,
        fmt_max,
    )


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
    kernel launch instead of two. See the module comment above for the
    call-site scoping rationale and why this stays a 2-state fusion rather
    than a whole-step batch.

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
            "update_scale_pair: Block1D/Block2D (FP4) scaling is"
            " unimplemented — PerTensor (fp8) only"
        )
