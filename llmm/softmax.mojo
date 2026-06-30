import compiler
from std.sys import simd_width_of
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
from std.gpu.primitives import block
from std.math import fma, ceildiv, exp
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Softmax Forward
# ===----------------------------------------------------------------------=== #


@always_inline
def _softmax_comp_max[
    dtype: DType,
    width: Int,
](
    idx: Int,
    logits_ptr: ImmutKernelPtr[dtype],
    mut s: SIMD[DType.float32, width],
    mut m: SIMD[DType.float32, width],
) -> None:
    var x = (logits_ptr + idx).load[width=width]().cast[DType.float32]()
    var new_m = max(m, x)
    s = s * exp(m - new_m) + exp(x - new_m)
    m = new_m


@always_inline
def softmax_phase_1_and_2_cpu[
    dtype: DType, width: Int
](
    idx: Int,
    logits_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,  # Our V
    vocab_size_padded: Int,  # Our Vp, padding is garbage
) -> Tuple[Scalar[DType.float32], Scalar[DType.float32]]:
    """Phase 1 (fused online max/sum over V) and phase 2 (lane merge) of
    the row softmax. Returns (m_row, s_row), ready for any epilogue:
    normalize, loss, or the fused classifier gradient."""
    var m = SIMD[DType.float32, width](Scalar[DType.float32].MIN_FINITE)
    var s = SIMD[DType.float32, width](0.0)

    # Fused Max with Sum over V. The while form keeps a single `v` (a `for`
    # loop variable would shadow it) and stops at the last full vector, so
    # the wide load never reads the garbage padding past V.
    var v = 0
    while v + width <= vocab_size:
        var new_idx = idx * vocab_size_padded + v
        _softmax_comp_max[dtype, width](new_idx, logits_ptr, s, m)
        v += width

    # Merge the lanes before the tail so the last V % width elements can
    # continue the recurrence on the merged scalars.
    var m_row = m.reduce_max()
    var s_row = (s * exp(m - SIMD[DType.float32, width](m_row))).reduce_add()

    # The tail of simd computations are handled here.
    for i in range(v, vocab_size):
        var new_idx = idx * vocab_size_padded + i
        _softmax_comp_max[dtype, 1](new_idx, logits_ptr, s_row, m_row)

    return (m_row, s_row)


@always_inline
def _softmax_fwd_cpu[
    dtype: DType, width: Int
](
    idx: Int,
    probs_ptr: MutKernelPtr[dtype],
    logits_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,  # Our V
    vocab_size_padded: Int,  # Our Vp, padding is garbage
) -> None:
    var stats = softmax_phase_1_and_2_cpu[dtype, width](
        idx, logits_ptr, vocab_size, vocab_size_padded
    )
    var m_row = stats[0]
    var s_row = stats[1]
    var inv_s = Scalar[DType.float32](1) / s_row

    @always_inline
    def _normalize[
        w: Int,
    ](local: Int) {
        probs_ptr,
        logits_ptr,
        m_row,
        inv_s,
        idx,
        vocab_size_padded,
    }:
        # vectorize advances `local` by w, so this body must handle w
        # elements with wide loads/stores; the broadcasts are built at w so
        # the tail instantiations (w < width) stay correct.
        var base = idx * vocab_size_padded + local
        var x = (logits_ptr + base).load[width=w]().cast[DType.float32]()
        var p = exp(x - SIMD[DType.float32, w](m_row)) * SIMD[DType.float32, w](
            inv_s
        )
        (probs_ptr + base).store[width=w](p.cast[dtype]())

    vectorize[width, unroll_factor=UNROLL](vocab_size, _normalize)


def softmax_fwd_cpu[
    dtype: DType,
    width: Int,
](
    probs_ptr: MutKernelPtr[dtype],
    logits_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) raises -> None:
    var total = Int(batch_size * seq_len)
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(total, max_workers)
    var num_workers = ceildiv(total, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var base = w * rows_per_worker
        var count = min(rows_per_worker, total - base)
        for local in range(count):
            var idx = base + local
            _softmax_fwd_cpu[dtype, width](
                idx,
                probs_ptr,
                logits_ptr,
                Int(vocab_size),
                Int(vocab_size_padded),
            )

    traced_parallelize["softmax_fwd", _worker](num_workers)


@always_inline
def softmax_phase_1_and_2_gpu[
    dtype: DType, BLOCK_SIZE: Int, width: Int = 4
](
    row: Int,
    tid: Int,
    logits_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,
    vocab_size_padded: Int,
) -> Tuple[Scalar[DType.float32], Scalar[DType.float32]]:
    """Phase 1 (per-thread online max/sum over the block-strided row) and
    phase 2 (per-thread merge, then block merge) of the row softmax.
    Returns (m_row, s_row) broadcast to every thread of the block.

    NOTE: contains block.max/block.sum, which synchronize the whole
    block, so every thread must call this the same number of times."""
    comptime BLOCK_SPAN = BLOCK_SIZE * width

    var m = SIMD[DType.float32, width](Scalar[DType.float32].MIN_FINITE)
    var s = SIMD[DType.float32, width](0.0)
    var m_tail = Scalar[DType.float32].MIN_FINITE
    var s_tail = Scalar[DType.float32](0.0)

    for tile_base in range(0, vocab_size, BLOCK_SPAN):
        var lane_base = tile_base + tid * width
        if lane_base + width <= vocab_size:
            var idx = row * vocab_size_padded + lane_base
            _softmax_comp_max[dtype, width](idx, logits_ptr, s, m)
        elif lane_base < vocab_size:
            # Ragged edge of the last tile.
            # This is so every thread keeps the same tile trip count for the reductions.
            for i in range(lane_base, vocab_size):
                var idx = row * vocab_size_padded + i
                _softmax_comp_max[dtype, 1](idx, logits_ptr, s_tail, m_tail)

    # Per-thread merge
    var m_thread = max(m.reduce_max(), m_tail)
    var s_thread = (
        s * exp(m - SIMD[DType.float32, width](m_thread))
    ).reduce_add() + s_tail * exp(m_tail - m_thread)

    # Merge across the block
    var m_row = block.max[block_size=BLOCK_SIZE](m_thread)
    s_thread = s_thread * exp(m_thread - m_row)
    var s_row = block.sum[block_size=BLOCK_SIZE](s_thread)

    return (m_row, s_row)


@always_inline
def _softmax_fwd_gpu[
    dtype: DType, BLOCK_SIZE: Int, width: Int = 4
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    probs_ptr: MutKernelPtr[dtype],
    logits_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,
    vocab_size_padded: Int,
) -> None:
    # NOTE: Thanks to the Modular team's implementation that I used as a reference, I was able to get this working faster.
    comptime BLOCK_SPAN = BLOCK_SIZE * width

    # Grid strided over the rows of the dispatched batch.
    # Each block walks rows at a stride of grid_dim.x until the rows are exhausted.
    # No need for a check that row < num_rows because the grid is bound to the range.
    # NOTE: the phase 1+2 helper synchronizes the whole block, so every
    # thread must take the same trip count through this loop.
    for row in range(block_row, num_rows, stride):
        var stats = softmax_phase_1_and_2_gpu[dtype, BLOCK_SIZE, width](
            row, tid, logits_ptr, vocab_size, vocab_size_padded
        )
        var m_row = stats[0]
        var s_row = stats[1]

        # Normalize the row with the same tiling, recomputing exp instead of
        # round-tripping the intermediate x - max through global memory.
        var inv_s = Scalar[DType.float32](1) / s_row
        var m_vec = SIMD[DType.float32, width](m_row)
        var inv_s_vec = SIMD[DType.float32, width](inv_s)
        for tile_base in range(0, vocab_size, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= vocab_size:
                var base = row * vocab_size_padded + lane_base
                var x = (
                    (logits_ptr + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                (probs_ptr + base).store[width=width](
                    (exp(x - m_vec) * inv_s_vec).cast[dtype]()
                )
            elif lane_base < vocab_size:
                for i in range(lane_base, vocab_size):
                    var x = logits_ptr[row * vocab_size_padded + i].cast[
                        DType.float32
                    ]()
                    probs_ptr[row * vocab_size_padded + i] = (
                        exp(x - m_row) * inv_s
                    ).cast[dtype]()


def softmax_fwd_gpu[
    dtype: DType, BLOCK_SIZE: Int, width: Int = 4
](
    probs_ptr: MutKernelPtr[dtype],
    logits_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    # NOTE: This is mainly for wrapping the dispatch parameters into a function.
    _softmax_fwd_gpu[dtype, BLOCK_SIZE, width](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        probs_ptr,
        logits_ptr,
        Int(vocab_size),
        Int(vocab_size_padded),
    )


def softmax_fwd[
    dtype: DType,
    target: StaticString,
](
    probs_ptr: MutKernelPtr[dtype],
    logits_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        softmax_fwd_cpu[dtype, simd_width](
            probs_ptr,
            logits_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = softmax_fwd_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            probs_ptr,
            logits_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("softmax_fwd")
struct SoftmaxFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        probs: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        logits: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if probs.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "probs must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        softmax_fwd[dtype, target](
            probs.unsafe_ptr(),
            logits.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Softmax Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _softmax_dot[
    dtype: DType,
    width: Int,
](
    idx: Int,
    probs_ptr: ImmutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    mut accumulator: SIMD[DType.float32, width],
) -> None:
    var d_prob = (d_probs_ptr + idx).load[width=width]().cast[DType.float32]()
    var prob = (probs_ptr + idx).load[width=width]().cast[DType.float32]()
    accumulator = fma(prob, d_prob, accumulator)


@always_inline
def _softmax_bwd_cpu[
    dtype: DType, width: Int
](
    idx: Int,
    d_logits_ptr: MutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    probs_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,  # Our V
    vocab_size_padded: Int,  # Our Vp, padding is garbage
) -> None:
    # The dot pass is a pure fma chain, so a single accumulator is
    # latency-bound (each fma waits on the previous one). A SIMD register
    # UNROLL times wider than the hardware lowers to UNROLL independent
    # registers and fma chains, filling the fma pipes; zero-init is safe
    # because an untouched lane contributes 0 to a sum.
    comptime unroll_width = width * UNROLL
    var accumulator = SIMD[DType.float32, unroll_width](0.0)
    var v = 0
    while v + unroll_width <= vocab_size:
        var new_idx = idx * vocab_size_padded + v
        _softmax_dot[dtype, unroll_width](
            new_idx, probs_ptr, d_probs_ptr, accumulator
        )
        v += unroll_width

    var dot = accumulator.reduce_add()

    # The tail (up to unroll_width - 1 elements) continues on the merged
    # scalar, exactly like the forward's tail recurrence.
    for i in range(v, vocab_size):
        var new_idx = idx * vocab_size_padded + i
        _softmax_dot[dtype, 1](new_idx, probs_ptr, d_probs_ptr, dot)

    @always_inline
    def _d_logits[
        w: Int,
    ](local: Int) {
        d_probs_ptr,
        probs_ptr,
        d_logits_ptr,
        dot,
        idx,
        vocab_size_padded,
    }:
        var base = idx * vocab_size_padded + local
        var p = (probs_ptr + base).load[width=w]().cast[DType.float32]()
        var g = (d_probs_ptr + base).load[width=w]().cast[DType.float32]()
        # NOTE: This could use an fma chain, but the compiler should be smart enough to do it.
        var d = p * (g - SIMD[DType.float32, w](dot))
        (d_logits_ptr + base).store[width=w](d.cast[dtype]())

    vectorize[width, unroll_factor=UNROLL](vocab_size, _d_logits)


def softmax_bwd_cpu[
    dtype: DType,
    width: Int,
](
    d_logits_ptr: MutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    probs_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) raises -> None:
    var total = Int(batch_size * seq_len)
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(total, max_workers)
    var num_workers = ceildiv(total, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var base = w * rows_per_worker
        var count = min(rows_per_worker, total - base)
        for local in range(count):
            var idx = base + local
            _softmax_bwd_cpu[dtype, width](
                idx,
                d_logits_ptr,
                d_probs_ptr,
                probs_ptr,
                Int(vocab_size),
                Int(vocab_size_padded),
            )

    traced_parallelize["softmax_bwd", _worker](num_workers)


@always_inline
def _softmax_bwd_gpu[
    dtype: DType, BLOCK_SIZE: Int, width: Int = 4
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    d_logits_ptr: MutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    probs_ptr: ImmutKernelPtr[dtype],
    vocab_size: Int,
    vocab_size_padded: Int,
) -> None:
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    for row in range(block_row, num_rows, stride):
        var accumulator = SIMD[DType.float32, width](0.0)
        var accumulator_tail = Scalar[DType.float32](0.0)

        for tile_base in range(0, vocab_size, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= vocab_size:
                var idx = row * vocab_size_padded + lane_base
                _softmax_dot[dtype, width](
                    idx, probs_ptr, d_probs_ptr, accumulator
                )
            elif lane_base < vocab_size:
                for i in range(lane_base, vocab_size):
                    var idx = row * vocab_size_padded + i
                    _softmax_dot[dtype, 1](
                        idx, probs_ptr, d_probs_ptr, accumulator_tail
                    )

        # Per-thread merge is plain adds (no max, no rescale), then ONE
        # block reduction. Inside the row loop: accumulators and dot_row
        # are per-row state, same scoping rule as the forward.
        var dot_thread = accumulator.reduce_add() + accumulator_tail
        var dot_row = block.sum[block_size=BLOCK_SIZE](dot_thread)

        var dot_vec = SIMD[DType.float32, width](dot_row)
        for tile_base in range(0, vocab_size, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= vocab_size:
                var base = row * vocab_size_padded + lane_base
                var p = (
                    (probs_ptr + base).load[width=width]().cast[DType.float32]()
                )
                var g = (
                    (d_probs_ptr + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var d = p * (g - dot_vec)
                (d_logits_ptr + base).store[width=width](d.cast[dtype]())
            elif lane_base < vocab_size:
                for i in range(lane_base, vocab_size):
                    var base = row * vocab_size_padded + i
                    var p = probs_ptr[base].cast[DType.float32]()
                    var g = d_probs_ptr[base].cast[DType.float32]()
                    var d = p * (g - dot_row)
                    d_logits_ptr[base] = d.cast[dtype]()


def softmax_bwd_gpu[
    dtype: DType, BLOCK_SIZE: Int, width: Int = 4
](
    d_logits_ptr: MutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    probs_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    # NOTE: This is mainly for wrapping the dispatch parameters into a function.
    _softmax_bwd_gpu[dtype, BLOCK_SIZE, width](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        d_logits_ptr,
        d_probs_ptr,
        probs_ptr,
        Int(vocab_size),
        Int(vocab_size_padded),
    )


def softmax_bwd[
    dtype: DType,
    target: StaticString,
](
    d_logits_ptr: MutKernelPtr[dtype],
    d_probs_ptr: ImmutKernelPtr[dtype],
    probs_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        softmax_bwd_cpu[dtype, simd_width](
            d_logits_ptr,
            d_probs_ptr,
            probs_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = softmax_bwd_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            d_logits_ptr,
            d_probs_ptr,
            probs_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("softmax_bwd")
struct SoftmaxBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_logits: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        d_probs: InputTensor[dtype=dtype, rank=1, static_spec=...],
        probs: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if d_logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "d_logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if d_probs.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "d_probs must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if probs.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "probs must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        softmax_bwd[dtype, target](
            d_logits.unsafe_ptr(),
            d_probs.unsafe_ptr(),
            probs.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )
