import compiler
from std.sys import simd_width_of
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.math import ceildiv, exp, log
from std.gpu.host import DeviceAttribute
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import barrier, block_idx, grid_dim, thread_idx

from llmm.softmax import softmax_phase_1_and_2_cpu, softmax_phase_1_and_2_gpu
from llmm.memory import ImmutKernelPtr, MutKernelPtr

# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Fused Classifier Forward and Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _fused_classifier_cpu[
    dtype: DType, width: Int, write_d_logits: Bool = True
](
    idx: Int,
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    vocab_size: Int,  # Our V
    vocab_size_padded: Int,  # Our Vp, padding is garbage
) -> None:
    var base = idx * vocab_size_padded
    var target_idx = Int(targets_ptr[idx])

    var stats = softmax_phase_1_and_2_cpu[dtype, width](
        idx, logits_ptr, vocab_size, vocab_size_padded
    )
    var m_row = stats[0]
    var s_row = stats[1]

    # NOTE: Loss in log-softmax form, and BEFORE the in-place overwrite below.
    var x_t = logits_ptr[base + target_idx].cast[DType.float32]()
    losses_ptr[idx] = log(s_row) + m_row - x_t

    comptime if write_d_logits:
        var d_loss = d_losses_ptr[idx]
        var inv_s = Scalar[DType.float32](1) / s_row

        @always_inline
        def _d_logits[
            w: Int,
        ](local: Int) {logits_ptr, m_row, inv_s, d_loss, base,}:
            # Indicator-free body: the one-hot touches exactly one element
            # of the row, fixed up scalarly after the loop. The in-place
            # wide load then store to the same location is safe
            # sequentially.
            var p_idx = base + local
            var x = (logits_ptr + p_idx).load[width=w]().cast[DType.float32]()
            var p = exp(x - SIMD[DType.float32, w](m_row)) * SIMD[
                DType.float32, w
            ](inv_s)
            (logits_ptr + p_idx).store[width=w](
                (p * SIMD[DType.float32, w](d_loss)).cast[dtype]()
            )

        vectorize[width, unroll_factor=UNROLL](vocab_size, _d_logits)

        # x_t is saved above, so we can compute the one-hot fix op here.
        var p_t = exp(x_t - m_row) * inv_s
        logits_ptr[base + target_idx] = ((p_t - 1.0) * d_loss).cast[dtype]()

        # Zero the padded tail so the backward matmul reads zeros.
        # NOTE: This is required for our kernel.
        for i in range(vocab_size, vocab_size_padded):
            logits_ptr[base + i] = Scalar[dtype](0)


def fused_classifier_cpu[
    dtype: DType,
    width: Int,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    var total = Int(batch_size * seq_len)
    var num_workers = min(total, parallelism_level())
    var rows_per_worker = ceildiv(total, num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * rows_per_worker
        var count = min(rows_per_worker, total - base)
        for local in range(count):
            var idx = base + local
            _fused_classifier_cpu[dtype, width, write_d_logits](
                idx,
                logits_ptr,
                losses_ptr,
                d_losses_ptr,
                targets_ptr,
                Int(vocab_size),
                Int(vocab_size_padded),
            )

    sync_parallelize[_worker](num_workers)


@always_inline
def _fused_classifier_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    write_d_logits: Bool = True,
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    vocab_size: Int,
    vocab_size_padded: Int,
) -> None:
    comptime BLOCK_SPAN = BLOCK_SIZE * width

    # Grid strided over the rows, same as the softmax kernels. The phase
    # 1+2 helper and the barrier below synchronize the whole block, so
    # every thread must take the same trip.
    for row in range(block_row, num_rows, stride):
        var base = row * vocab_size_padded
        var target_idx = Int(targets_ptr[row])

        var stats = softmax_phase_1_and_2_gpu[dtype, BLOCK_SIZE, width](
            row, tid, logits_ptr, vocab_size, vocab_size_padded
        )
        var m_row = stats[0]
        var s_row = stats[1]

        # Loss in log-softmax form, single thread. NOTE: assigns, where
        # llm.c accumulates with -=; this matches crossentropy_ohe_fwd.
        if tid == 0:
            var x_t = logits_ptr[base + target_idx].cast[DType.float32]()
            losses_ptr[row] = log(s_row) + m_row - x_t

        comptime if write_d_logits:
            # NOTE: Same as _syncronize() in Karpathy's kernel
            barrier()

            var d_loss = d_losses_ptr[row]
            var inv_s = Scalar[DType.float32](1) / s_row
            var m_vec = SIMD[DType.float32, width](m_row)
            var inv_s_vec = SIMD[DType.float32, width](inv_s)
            var d_loss_vec = SIMD[DType.float32, width](d_loss)

            for tile_base in range(0, vocab_size, BLOCK_SPAN):
                var lane_base = tile_base + tid * width
                if lane_base + width <= vocab_size:
                    var x = (
                        (logits_ptr + base + lane_base)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    var p = exp(x - m_vec) * inv_s_vec
                    var d = p * d_loss_vec
                    if (
                        lane_base <= target_idx
                        and target_idx < lane_base + width
                    ):
                        var k = target_idx - lane_base
                        d[k] = d[k] - d_loss
                    (logits_ptr + base + lane_base).store[width=width](
                        d.cast[dtype]()
                    )
                elif lane_base < vocab_size:
                    # Ragged edge of the last tile: scalar steps, same
                    # uniform-trip-count rule as the softmax kernels.
                    for i in range(lane_base, vocab_size):
                        var x = logits_ptr[base + i].cast[DType.float32]()
                        var p = exp(x - m_row) * inv_s
                        var ind = Scalar[DType.float32](
                            1.0
                        ) if i == target_idx else Scalar[DType.float32](0.0)
                        logits_ptr[base + i] = ((p - ind) * d_loss).cast[
                            dtype
                        ]()

            # Zero the padded tail so the backward matmul reads zeros.
            for i in range(vocab_size + tid, vocab_size_padded, BLOCK_SIZE):
                logits_ptr[base + i] = Scalar[dtype](0)


def fused_classifier_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    _fused_classifier_gpu[dtype, BLOCK_SIZE, width, write_d_logits](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        logits_ptr,
        losses_ptr,
        d_losses_ptr,
        targets_ptr,
        Int(vocab_size),
        Int(vocab_size_padded),
    )


def fused_classifier[
    dtype: DType,
    target: StaticString,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        fused_classifier_cpu[dtype, simd_width, write_d_logits](
            logits_ptr,
            losses_ptr,
            d_losses_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        # Duplicated gpu dispatch code from the softmax ops.
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = fused_classifier_gpu[
            dtype, BLOCK_SIZE, write_d_logits=write_d_logits
        ]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            logits_ptr,
            losses_ptr,
            d_losses_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# Fused Classifier Compiler Registration
# ===----------------------------------------------------------------------=== #


@compiler.register("fused_classifier")
struct FusedClassifier:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        logits: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        losses: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_losses: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if vocab_size > vocab_size_padded:
            raise Error("vocab_size must not exceed vocab_size_padded")
        if logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if losses.size() != Int(batch_size * seq_len):
            raise Error(
                "losses must have the same size as batch_size * seq_len"
            )
        if d_losses.size() != Int(batch_size * seq_len):
            raise Error(
                "d_losses must have the same size as batch_size * seq_len"
            )
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        fused_classifier[dtype, target, True](
            logits.unsafe_ptr(),
            losses.unsafe_ptr(),
            d_losses.unsafe_ptr(),
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )


@compiler.register("fused_classifier_fwd")
struct FusedClassifierFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        losses: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        logits: InputTensor[dtype=dtype, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if vocab_size > vocab_size_padded:
            raise Error("vocab_size must not exceed vocab_size_padded")
        if logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if losses.size() != Int(batch_size * seq_len):
            raise Error(
                "losses must have the same size as batch_size * seq_len"
            )
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        # The write_d_logits=False instantiation contains no stores to
        # logits (comptime-dead code), so handing the shared kernel
        # signature a mutable-origin pointer is sound. The dangling
        # d_losses sentinel is likewise never dereferenced.
        var null_d_losses = ImmutKernelPtr[DType.float32].unsafe_dangling()
        fused_classifier[dtype, target, False](
            logits.unsafe_ptr(),
            losses.unsafe_ptr(),
            null_d_losses,
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )
