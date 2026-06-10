import compiler
from tensor import InputTensor
from std.sys import simd_width_of
from std.memory import UnsafePointer
from std.math import fma, sqrt, ceildiv, log
from std.gpu.host.info import is_cpu, is_gpu
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu import block_dim, block_idx, thread_idx
from std.algorithm import vectorize, sync_parallelize
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime CHUNK_SIZE = 4096
comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Cross-entropy Loss One-Hot Encoding Forward
# ===----------------------------------------------------------------------=== #


def _crossentropy_ohe_fwd[
    dtype: DType,
](
    idx: Int,
    losses_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    # Load the target at the current index as an int64 which is the index of the target token in the vocabulary.
    var target_idx = Int(targets_ptr[idx])
    # Load the probability at the target vocab index. Cast to float32 to take the log
    # This is equal to setting the pointer offset to b * T * Vp + t * Vp in the iterative loop.
    # This is because you can arrange it as Vp * (b * T + t) which in our thread parallized loop is equal to idx
    # Then when we look at the target_idx, we are just adding that as the offset. (Similar to the ptr arithmitc aboe)
    # Int(vocab_size_padded) is load-bearing: without it the offset math
    # falls back to a deprecated implicit Int -> Int64 conversion.
    var prob = probs_ptr[idx * Int(vocab_size_padded) + target_idx].cast[
        DType.float32
    ]()
    # Store the log of the probability to the losses pointer.
    losses_ptr[idx] = -log(prob)


def crossentropy_ohe_fwd_cpu[
    dtype: DType,
](
    losses_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    # NOTE: This function is uniquely different from the standard cross-entropu loss calculation.
    # Instead of neeeding to sum over the full vocabulary, we can just use the one-hot nature of
    # the probabilities to calculate the loss of the targets where softmax is 1.0 and the rest are 0.0.
    # This means we can calculate the loss in a single vectorized operation

    var num_chunks = Int((batch_size * seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)

    @parameter
    def _chunk(c: Int):
        var base = c * CHUNK_SIZE
        var total = Int(batch_size * seq_len)  # Our B * T
        var count = min(
            CHUNK_SIZE, total - base
        )  # Used to count our local iterator for this thread.

        # We can't improve this over the SIMD width because we aren't loading over a single memory location.
        # Karpathy's logic uses b * T + t as the index to the target tokens.
        # Because we are using synchronous parallelization, we can use the local index to calculate the index to the target tokens.
        # This way we compute over many chunks in parallel.
        for local in range(count):
            var idx = base + local
            _crossentropy_ohe_fwd[dtype](
                idx,
                losses_ptr,
                probs_ptr,
                targets_ptr,
                batch_size,
                seq_len,
                vocab_size_padded,
            )

    sync_parallelize[_chunk](num_chunks)


def crossentropy_ohe_fwd_gpu[
    dtype: DType,
](
    losses_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < Int(batch_size * seq_len):
        _crossentropy_ohe_fwd[dtype](
            idx,
            losses_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
        )


def crossentropy_ohe_fwd[
    dtype: DType,
    target: StaticString,
](
    losses_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContextPtr,
) capturing raises:
    comptime if is_cpu[target]():
        crossentropy_ohe_fwd_cpu[dtype](
            losses_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        var dev_ctx = ctx.get_device_context()
        # Each thread handles the a batch_size * seq_len element. Unlike our adamw op that also handles width elements.
        # One GPU thread per (b, t) output — no SIMD width, so num_threads = B*T.
        var num_threads = Int(batch_size * seq_len)
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)

        comptime gpu_kernel = crossentropy_ohe_fwd_gpu[dtype]
        var compiled = dev_ctx.compile_function[
            func=gpu_kernel, signature_func=gpu_kernel
        ]()
        dev_ctx.enqueue_function(
            compiled,
            losses_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("crossentropy_ohe_fwd")
struct CrossEntropyOHEFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        losses: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        probs: InputTensor[dtype=dtype, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        # MAX binds runtime scalars to Scalar[...] params only (a plain Int
        # is rejected at graph load), hence Int64 shape args.
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContextPtr,
    ) capturing raises:
        if losses.size() != Int(batch_size * seq_len):
            raise Error(
                "losses must have the same size as batch_size * seq_len"
            )
        if probs.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "probs must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        crossentropy_ohe_fwd[dtype, target](
            losses.unsafe_ptr(),
            probs.unsafe_ptr(),
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size_padded,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Cross-entropy Loss One-Hot Encoding Backward
# ===----------------------------------------------------------------------=== #


def _crossentropy_ohe_bwd[
    dtype: DType,
](
    idx: Int,
    d_losses_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_probs_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp — only used as row stride here
) -> None:
    var target_idx = Int(targets_ptr[idx])
    var target_offset = idx * Int(vocab_size_padded) + target_idx
    var prob = probs_ptr[target_offset].cast[DType.float32]()
    var d_loss = d_losses_ptr[idx].cast[DType.float32]()
    var d_prob = -d_loss / prob
    d_probs_ptr[target_offset] = d_prob.cast[dtype]()


def crossentropy_ohe_bwd_cpu[
    dtype: DType,
](
    d_losses_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_probs_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    var num_chunks = Int((batch_size * seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)

    @parameter
    def _chunk(c: Int):
        var base = c * CHUNK_SIZE
        var total = Int(batch_size * seq_len)
        var count = min(CHUNK_SIZE, total - base)

        for local in range(count):
            var idx = base + local
            _crossentropy_ohe_bwd[dtype](
                idx,
                d_losses_ptr,
                d_probs_ptr,
                probs_ptr,
                targets_ptr,
                batch_size,
                seq_len,
                vocab_size_padded,
            )

    sync_parallelize[_chunk](num_chunks)


def crossentropy_ohe_bwd_gpu[
    dtype: DType,
](
    d_losses_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_probs_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < Int(batch_size * seq_len):
        _crossentropy_ohe_bwd[dtype](
            idx,
            d_losses_ptr,
            d_probs_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
        )


def crossentropy_ohe_bwd[
    dtype: DType,
    target: StaticString,
](
    d_losses_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_probs_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    probs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    targets_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContextPtr,
) capturing raises:
    comptime if is_cpu[target]():
        crossentropy_ohe_bwd_cpu[dtype](
            d_losses_ptr,
            d_probs_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        # Duplicated gpu dispatch code from the fwd op.
        comptime BLOCK_SIZE = 256
        var dev_ctx = ctx.get_device_context()
        # One GPU thread per (b, t) output — no SIMD width, so num_threads = B*T.
        var num_threads = Int(batch_size * seq_len)
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)
        comptime gpu_kernel = crossentropy_ohe_bwd_gpu[dtype]
        var compiled = dev_ctx.compile_function[
            func=gpu_kernel, signature_func=gpu_kernel
        ]()
        dev_ctx.enqueue_function(
            compiled,
            d_losses_ptr,
            d_probs_ptr,
            probs_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("crossentropy_ohe_bwd")
struct CrossEntropyOHEBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_losses: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        d_probs: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        probs: InputTensor[dtype=dtype, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        # MAX binds runtime scalars to Scalar[...] params only (a plain Int
        # is rejected at graph load), hence Int64 shape args.
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContextPtr,
    ) capturing raises:
        if d_losses.size() != Int(batch_size * seq_len):
            raise Error(
                "d_losses must have the same size as batch_size * seq_len"
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
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        crossentropy_ohe_bwd[dtype, target](
            d_losses.unsafe_ptr(),
            d_probs.unsafe_ptr(),
            probs.unsafe_ptr(),
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size_padded,
            ctx,
        )
