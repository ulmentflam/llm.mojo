import compiler
from std.memory import alloc
from std.sys import simd_width_of, align_of
from std.gpu.primitives import block, warp
from std.gpu import WARP_SIZE
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
from std.collections import InlineArray
from std.math import fma, ceildiv, rsqrt
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import sync_parallelize, vectorize
from std.gpu import block_dim, block_idx, grid_dim, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.atomic import Atomic
from layout import Layout, LayoutTensor

from llmm.profiler import traced_parallelize
from llmm.memory import (
    ImmutKernelPtr,
    MutKernelPtr,
    persistent_device_buffer,
)

# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4

# ===----------------------------------------------------------------------=== #
# LayerNorm Forward
# ===----------------------------------------------------------------------=== #


@always_inline
def _layernorm_fwd_cpu[
    dtype: DType,
    width: Int,
](
    idx: Int,
    output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    channels: Int,
) -> None:
    var row_offset = idx * channels

    # Phase 1: Compute the mean.
    var i = 0
    var sum_vec = SIMD[DType.float32, width](0.0)
    while i + width <= channels:
        sum_vec += (
            (input_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        i += width
    var mean = sum_vec.reduce_add()

    for j in range(i, channels):
        mean += (
            (input_ptr + row_offset + j).load[width=1]().cast[DType.float32]()
        )

    mean /= Float32(channels)

    # Phase 2: Compute the variance.
    i = 0
    var var_vec = SIMD[DType.float32, width](0.0)
    var mean_vec = SIMD[DType.float32, width](mean)
    while i + width <= channels:
        var x = (
            (input_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var diff = x - mean_vec
        var_vec = fma(diff, diff, var_vec)
        i += width
    var variance = var_vec.reduce_add()

    for j in range(i, channels):
        var x = (
            (input_ptr + row_offset + j).load[width=1]().cast[DType.float32]()
        )
        var diff = x - mean
        variance = fma(diff, diff, variance)
    variance /= Float32(channels)

    var rstd = rsqrt(variance + epsilon)

    # Phase 3: Normalize and Apply weights.
    i = 0
    var rstd_vec = SIMD[DType.float32, width](rstd)
    while i + width <= channels:
        var x = (
            (input_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var g = (gamma_ptr + i).load[width=width]().cast[DType.float32]()
        var b = (beta_ptr + i).load[width=width]().cast[DType.float32]()

        var n = rstd_vec * (x - mean_vec)
        var o = fma(n, g, b)
        (output_ptr + row_offset + i).store(o.cast[dtype]())
        i += width

    for j in range(i, channels):
        var x = (
            (input_ptr + row_offset + j).load[width=1]().cast[DType.float32]()
        )
        var g = gamma_ptr[j].cast[DType.float32]()
        var b = beta_ptr[j].cast[DType.float32]()

        var n = rstd * (x - mean)
        var o = n * g + b
        (output_ptr + row_offset + j).store(o.cast[dtype]())

    # Store the Mean and RSTD for the backward pass.
    mean_ptr[idx] = mean
    rstd_ptr[idx] = rstd


def layernorm_fwd_cpu[
    dtype: DType,
    width: Int,
](
    output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
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
            _layernorm_fwd_cpu[dtype, width](
                idx,
                output_ptr,
                input_ptr,
                gamma_ptr,
                beta_ptr,
                epsilon,
                mean_ptr,
                rstd_ptr,
                Int(channels),
            )

    traced_parallelize["layernorm_fwd", _worker](num_workers)


@always_inline
def _layernorm_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    channels: Int,
) -> None:
    comptime BLOCK_SPAN = BLOCK_SIZE * width

    for row in range(block_row, num_rows, stride):
        # Pass 1: Mean
        var sum_thread = SIMD[DType.float32, width](0.0)
        var sum_tail = Scalar[DType.float32](0.0)

        for tile_base in range(0, channels, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= channels:
                sum_thread += (
                    (input_ptr + row * channels + lane_base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
            elif lane_base < channels:
                for i in range(lane_base, channels):
                    sum_tail += (
                        (input_ptr + row * channels + i)
                        .load[width=1]()
                        .cast[DType.float32]()
                    )

        var mean_thread = sum_thread.reduce_add() + sum_tail
        var mean = block.sum[block_size=BLOCK_SIZE](mean_thread) / Float32(
            channels
        )

        # Pass 2: Variance
        var var_thread = SIMD[DType.float32, width](0.0)
        var var_tail = Scalar[DType.float32](0.0)
        var mean_vec = SIMD[DType.float32, width](mean)
        for tile_base in range(0, channels, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= channels:
                var x = (
                    (input_ptr + row * channels + lane_base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var diff = x - mean_vec
                var_thread = fma(diff, diff, var_thread)
            elif lane_base < channels:
                for i in range(lane_base, channels):
                    var x = (
                        (input_ptr + row * channels + i)
                        .load[width=1]()
                        .cast[DType.float32]()
                    )
                    var diff = x - mean
                    var_tail = fma(diff, diff, var_tail)

        var variance_thread = var_thread.reduce_add() + var_tail
        var variance = block.sum[block_size=BLOCK_SIZE](
            variance_thread
        ) / Float32(channels)
        var rstd = rsqrt(variance + epsilon)

        # Pass 3: Output
        var rstd_vec = SIMD[DType.float32, width](rstd)
        for tile_base in range(0, channels, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= channels:
                var idx = row * channels + lane_base
                var x = (
                    (input_ptr + idx).load[width=width]().cast[DType.float32]()
                )
                var g = (
                    (gamma_ptr + lane_base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var b = (
                    (beta_ptr + lane_base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var n = rstd_vec * (x - mean_vec)
                var o = fma(n, g, b)
                (output_ptr + idx).store(o.cast[dtype]())
            elif lane_base < channels:
                for i in range(lane_base, channels):
                    var idx = row * channels + i
                    var x = (
                        (input_ptr + idx).load[width=1]().cast[DType.float32]()
                    )
                    var g = gamma_ptr[i].cast[DType.float32]()
                    var b = beta_ptr[i].cast[DType.float32]()
                    var n = rstd * (x - mean)
                    var o = n * g + b
                    (output_ptr + idx).store(o.cast[dtype]())

        if tid == 0:
            mean_ptr[row] = mean
            rstd_ptr[row] = rstd


def layernorm_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
](
    output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    _layernorm_fwd_gpu[dtype, BLOCK_SIZE, width](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        output_ptr,
        input_ptr,
        gamma_ptr,
        beta_ptr,
        epsilon,
        mean_ptr,
        rstd_ptr,
        Int(channels),
    )


def layernorm_fwd[
    dtype: DType,
    target: StaticString,
](
    output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        layernorm_fwd_cpu[dtype, width](
            output_ptr,
            input_ptr,
            gamma_ptr,
            beta_ptr,
            epsilon,
            mean_ptr,
            rstd_ptr,
            batch_size,
            seq_len,
            channels,
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

        comptime gpu_kernel = layernorm_fwd_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            output_ptr,
            input_ptr,
            gamma_ptr,
            beta_ptr,
            epsilon,
            mean_ptr,
            rstd_ptr,
            batch_size,
            seq_len,
            channels,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("layernorm_fwd")
struct LayerNormFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        gamma: InputTensor[dtype=dtype, rank=1, static_spec=...],
        beta: InputTensor[dtype=dtype, rank=1, static_spec=...],
        epsilon: Scalar[DType.float32],
        mean: MutableInputTensor[dtype=DType.float32, rank=1, static_spec=...],
        rstd: MutableInputTensor[dtype=DType.float32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        channels: Int64,  # Our C
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != Int(batch_size * seq_len * channels):
            raise Error("output size mismatch")
        if x.size() != Int(batch_size * seq_len * channels):
            raise Error("x size mismatch")
        if gamma.size() != Int(channels):
            raise Error("gamma size mismatch")
        if beta.size() != Int(channels):
            raise Error("beta size mismatch")
        if mean.size() != Int(batch_size * seq_len):
            raise Error("mean size mismatch")
        if rstd.size() != Int(batch_size * seq_len):
            raise Error("rstd size mismatch")

        layernorm_fwd[dtype, target](
            output.unsafe_ptr(),
            x.unsafe_ptr(),
            gamma.unsafe_ptr(),
            beta.unsafe_ptr(),
            epsilon,
            mean.unsafe_ptr(),
            rstd.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# LayerNorm Fused Residual Forward
# ===----------------------------------------------------------------------=== #


@always_inline
def _layernorm_fused_residual_fwd_cpu[
    dtype: DType,
    width: Int,
](
    idx: Int,
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    channels: Int,
) -> None:
    var row_offset = idx * channels

    # Phase 1: Compute the mean and write to residual.
    var i = 0
    var sum_vec = SIMD[DType.float32, width](0.0)
    while i + width <= channels:
        var val1 = (inp1_ptr + row_offset + i).load[width=width]()
        var val2 = (inp2_ptr + row_offset + i).load[width=width]()
        var sum_val = val1 + val2
        (residual_ptr + row_offset + i).store(sum_val)
        sum_vec += sum_val.cast[DType.float32]()
        i += width
    var mean = sum_vec.reduce_add()

    for j in range(i, channels):
        var val1 = (inp1_ptr + row_offset + j).load[width=1]()
        var val2 = (inp2_ptr + row_offset + j).load[width=1]()
        var sum_val = val1 + val2
        (residual_ptr + row_offset + j).store(sum_val)
        mean += sum_val.cast[DType.float32]()

    mean /= Float32(channels)

    # Phase 2: Compute the variance.
    i = 0
    var var_vec = SIMD[DType.float32, width](0.0)
    var mean_vec = SIMD[DType.float32, width](mean)
    while i + width <= channels:
        var x = (
            (residual_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var diff = x - mean_vec
        var_vec = fma(diff, diff, var_vec)
        i += width
    var variance = var_vec.reduce_add()

    for j in range(i, channels):
        var x = (
            (residual_ptr + row_offset + j)
            .load[width=1]()
            .cast[DType.float32]()
        )
        var diff = x - mean
        variance = fma(diff, diff, variance)
    variance /= Float32(channels)

    var rstd = rsqrt(variance + epsilon)

    # Phase 3: Normalize and Apply weights.
    i = 0
    var rstd_vec = SIMD[DType.float32, width](rstd)
    while i + width <= channels:
        var x = (
            (residual_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var g = (gamma_ptr + i).load[width=width]().cast[DType.float32]()
        var b = (beta_ptr + i).load[width=width]().cast[DType.float32]()

        var n = rstd_vec * (x - mean_vec)
        var o = fma(n, g, b)
        (normed_ptr + row_offset + i).store(o.cast[dtype]())
        i += width

    for j in range(i, channels):
        var x = (
            (residual_ptr + row_offset + j)
            .load[width=1]()
            .cast[DType.float32]()
        )
        var g = gamma_ptr[j].cast[DType.float32]()
        var b = beta_ptr[j].cast[DType.float32]()

        var n = rstd * (x - mean)
        var o = n * g + b
        (normed_ptr + row_offset + j).store(o.cast[dtype]())

    # Store the Mean and RSTD for the backward pass.
    mean_ptr[idx] = mean
    rstd_ptr[idx] = rstd


def layernorm_fused_residual_fwd_cpu[
    dtype: DType,
    width: Int,
](
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
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
            _layernorm_fused_residual_fwd_cpu[dtype, width](
                idx,
                residual_ptr,
                normed_ptr,
                inp1_ptr,
                inp2_ptr,
                gamma_ptr,
                beta_ptr,
                epsilon,
                mean_ptr,
                rstd_ptr,
                Int(channels),
            )

    traced_parallelize["layernorm_fused_fwd", _worker](num_workers)


@always_inline
def _layernorm_fused_residual_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    channels: Int,
) -> None:
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    # Cache the residual row in shared so passes 2 (variance) and 3 (normalize)
    # read it from SMEM instead of re-reading global. CAP covers GPT-2's C=768;
    # a block handles ≤ CAP channels per row.
    comptime LN_CAP = 2048
    var s_res = LayoutTensor[
        DType.float32,
        Layout.row_major(LN_CAP),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # `aligned` (comptime): True keeps the vectorized fast path with explicit
    # 16-byte alignment on the global loads/stores (byte-identical to before)
    # for the production invariant channels % width == 0, where idx =
    # row*channels + lane_base is provably width-aligned. False (odd channel
    # counts, e.g. the equivalence suite's 767) falls back to a fully scalar
    # sweep — row*channels is not provably width-aligned there, and a forced
    # over-aligned SIMD access crashes with CUDA_ERROR_MISALIGNED_ADDRESS.
    comptime align = align_of[SIMD[dtype, width]]()

    for row in range(block_row, num_rows, stride):
        # Pass 1: Add input tensors, store to residual (global + shared), mean.
        var sum_thread = SIMD[DType.float32, width](0.0)
        var sum_tail = Scalar[DType.float32](0.0)

        comptime if aligned:
            for tile_base in range(0, channels, BLOCK_SPAN):
                var lane_base = tile_base + tid * width
                if lane_base + width <= channels:
                    var idx = row * channels + lane_base
                    var val1 = (inp1_ptr + idx).load[
                        width=width, alignment=align
                    ]()
                    var val2 = (inp2_ptr + idx).load[
                        width=width, alignment=align
                    ]()
                    var sum_val = val1 + val2
                    (residual_ptr + idx).store[alignment=align](sum_val)
                    var sv_f = sum_val.cast[DType.float32]()
                    (s_res.ptr + lane_base).store[width=width](sv_f)
                    sum_thread += sv_f
                elif lane_base < channels:
                    for i in range(lane_base, channels):
                        var idx = row * channels + i
                        var val1 = (inp1_ptr + idx).load[width=1]()
                        var val2 = (inp2_ptr + idx).load[width=1]()
                        var sum_val = val1 + val2
                        (residual_ptr + idx).store(sum_val)
                        var sv_f = sum_val.cast[DType.float32]()
                        s_res.ptr[i] = sv_f[0]
                        sum_tail += sv_f
        else:
            for i in range(tid, channels, BLOCK_SIZE):
                var idx = row * channels + i
                var val1 = inp1_ptr[idx]
                var val2 = inp2_ptr[idx]
                var sum_val = val1 + val2
                residual_ptr[idx] = sum_val
                var sv_f = sum_val.cast[DType.float32]()
                s_res.ptr[i] = sv_f
                sum_tail += sv_f
        barrier()

        var mean_thread = sum_thread.reduce_add() + sum_tail
        var mean = block.sum[block_size=BLOCK_SIZE](mean_thread) / Float32(
            channels
        )

        # Pass 2: Variance (reads residual from shared).
        var var_thread = SIMD[DType.float32, width](0.0)
        var var_tail = Scalar[DType.float32](0.0)
        var mean_vec = SIMD[DType.float32, width](mean)
        comptime if aligned:
            for tile_base in range(0, channels, BLOCK_SPAN):
                var lane_base = tile_base + tid * width
                if lane_base + width <= channels:
                    var x = (s_res.ptr + lane_base).load[width=width]()
                    var diff = x - mean_vec
                    var_thread = fma(diff, diff, var_thread)
                elif lane_base < channels:
                    for i in range(lane_base, channels):
                        var x = s_res.ptr[i]
                        var diff = x - mean
                        var_tail = fma(diff, diff, var_tail)
        else:
            for i in range(tid, channels, BLOCK_SIZE):
                var x = s_res.ptr[i]
                var diff = x - mean
                var_tail = fma(diff, diff, var_tail)

        var variance_thread = var_thread.reduce_add() + var_tail
        var variance = block.sum[block_size=BLOCK_SIZE](
            variance_thread
        ) / Float32(channels)
        var rstd = rsqrt(variance + epsilon)

        # Pass 3: Output (Normed) — read residual from shared.
        var rstd_vec = SIMD[DType.float32, width](rstd)
        comptime if aligned:
            for tile_base in range(0, channels, BLOCK_SPAN):
                var lane_base = tile_base + tid * width
                if lane_base + width <= channels:
                    var idx = row * channels + lane_base
                    var x = (s_res.ptr + lane_base).load[width=width]()
                    var g = (
                        (gamma_ptr + lane_base)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var b = (
                        (beta_ptr + lane_base)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var n = rstd_vec * (x - mean_vec)
                    var o = fma(n, g, b)
                    (normed_ptr + idx).store[alignment=align](o.cast[dtype]())
                elif lane_base < channels:
                    for i in range(lane_base, channels):
                        var idx = row * channels + i
                        var x = s_res.ptr[i]
                        var g = gamma_ptr[i].cast[DType.float32]()
                        var b = beta_ptr[i].cast[DType.float32]()
                        var n = rstd * (x - mean)
                        var o = n * g + b
                        (normed_ptr + idx).store(o.cast[dtype]())
        else:
            for i in range(tid, channels, BLOCK_SIZE):
                var idx = row * channels + i
                var x = s_res.ptr[i]
                var g = gamma_ptr[i].cast[DType.float32]()
                var b = beta_ptr[i].cast[DType.float32]()
                var n = rstd * (x - mean)
                var o = n * g + b
                normed_ptr[idx] = o.cast[dtype]()

        if tid == 0:
            mean_ptr[row] = mean
            rstd_ptr[row] = rstd
        # Fence before the next row reuses the shared residual buffer.
        barrier()


def _ln_fused_residual_warp_gpu[
    dtype: DType,
    WARPS: Int,
    width: Int,
](
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    num_rows: Int,
    channels: Int,
) -> None:
    # One WARP per row + warp-shuffle reductions (llm.c's fused_residual kernel5),
    # replacing one block/row + slow block.sum. gamma/beta cached in shared once
    # per block; the residual row cached in shared for passes 2/3. Assumes
    # channels % (WARP_SIZE*width) == 0 (holds for GPT-2 C=768, width=8).
    #
    # METAL HAZARD (dead code, not dispatched): this kernel allocates
    #   s_weight[CAP] + s_bias[CAP] + s_res[WARPS * CAP] shared-memory slots.
    # For dtype=float32 and WARPS=8: 1024*4*(2+8) = 40 960 bytes, which exceeds
    # Metal's 32 768-byte threadgroup-memory limit.
    # Do NOT dispatch this kernel on Apple GPU with float32.  Either reduce CAP
    # or switch to the production _layernorm_fused_residual_fwd_gpu path
    # (uses only s_res[LN_CAP=2048] = 8 192 bytes → safely within limit).
    comptime CAP = 1024
    var s_weight = LayoutTensor[
        dtype,
        Layout.row_major(CAP),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var s_bias = LayoutTensor[
        dtype,
        Layout.row_major(CAP),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var s_res = LayoutTensor[
        dtype,
        Layout.row_major(WARPS * CAP),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    # Cooperatively cache gamma/beta for this block's rows.
    var e = tid
    while e < channels:
        s_weight.ptr[e] = gamma_ptr[e]
        s_bias.ptr[e] = beta_ptr[e]
        e += WARPS * WARP_SIZE
    barrier()

    var row = Int(block_idx.x) * WARPS + warp_id
    if row >= num_rows:
        return
    var base = row * channels
    var res_warp = s_res.ptr + warp_id * CAP

    # Pass 1: residual = inp1+inp2 (→ global + shared), warp-sum for mean.
    var sum = Scalar[DType.float32](0.0)
    var c = lane * width
    while c < channels:
        var v = (inp1_ptr + base + c).load[width=width]() + (
            inp2_ptr + base + c
        ).load[width=width]()
        (residual_ptr + base + c).store(v)
        (res_warp + c).store[width=width](v)
        var vf = v.cast[DType.float32]()
        comptime for k in range(width):
            sum += vf[k]
        c += WARP_SIZE * width
    var m = warp.sum(sum) / Float32(channels)

    # Pass 2: variance from shared.
    var var_acc = Scalar[DType.float32](0.0)
    c = lane * width
    while c < channels:
        var res = (res_warp + c).load[width=width]().cast[DType.float32]()
        comptime for k in range(width):
            var d = res[k] - m
            var_acc += d * d
        c += WARP_SIZE * width
    var variance = warp.sum(var_acc) / Float32(channels)
    var rstd = rsqrt(variance + epsilon)

    # Pass 3: normalize from shared.
    c = lane * width
    while c < channels:
        var res = (res_warp + c).load[width=width]().cast[DType.float32]()
        var g = (s_weight.ptr + c).load[width=width]().cast[DType.float32]()
        var b = (s_bias.ptr + c).load[width=width]().cast[DType.float32]()
        var out = SIMD[dtype, width](0)
        comptime for k in range(width):
            out[k] = (rstd * (res[k] - m) * g[k] + b[k]).cast[dtype]()
        (normed_ptr + base + c).store(out)
        c += WARP_SIZE * width

    if lane == 0:
        mean_ptr[row] = m
        rstd_ptr[row] = rstd


def layernorm_fused_residual_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    _layernorm_fused_residual_fwd_gpu[
        dtype, BLOCK_SIZE, width, aligned=aligned
    ](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        residual_ptr,
        normed_ptr,
        inp1_ptr,
        inp2_ptr,
        gamma_ptr,
        beta_ptr,
        epsilon,
        mean_ptr,
        rstd_ptr,
        Int(channels),
    )


def layernorm_fused_residual_fwd[
    dtype: DType,
    target: StaticString,
](
    residual_ptr: MutKernelPtr[dtype],
    normed_ptr: MutKernelPtr[dtype],
    inp1_ptr: ImmutKernelPtr[dtype],
    inp2_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    beta_ptr: ImmutKernelPtr[dtype],
    epsilon: Scalar[DType.float32],
    mean_ptr: MutKernelPtr[DType.float32],
    rstd_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        layernorm_fused_residual_fwd_cpu[dtype, width](
            residual_ptr,
            normed_ptr,
            inp1_ptr,
            inp2_ptr,
            gamma_ptr,
            beta_ptr,
            epsilon,
            mean_ptr,
            rstd_ptr,
            batch_size,
            seq_len,
            channels,
        )
    elif is_gpu[target]():
        # (A warp-per-row port of llm.c's kernel5 was tried and measured SLOWER
        # in Mojo — 120 vs 95 µs/call — due to shared-mem occupancy + warp.sum
        # overhead; the block-per-row kernel is kept.)
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        comptime width = 4
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        # Dispatch aligned vs. scalar-fallback kernels at the host; see
        # _layernorm_fused_residual_fwd_gpu's docstring.
        if Int(channels) % width == 0:
            comptime gpu_kernel = layernorm_fused_residual_fwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=True
            ]
            var compiled = device_ctx.compile_function[gpu_kernel]()
            device_ctx.enqueue_function(
                compiled,
                residual_ptr,
                normed_ptr,
                inp1_ptr,
                inp2_ptr,
                gamma_ptr,
                beta_ptr,
                epsilon,
                mean_ptr,
                rstd_ptr,
                batch_size,
                seq_len,
                channels,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            comptime gpu_kernel_u = layernorm_fused_residual_fwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=False
            ]
            var compiled_u = device_ctx.compile_function[gpu_kernel_u]()
            device_ctx.enqueue_function(
                compiled_u,
                residual_ptr,
                normed_ptr,
                inp1_ptr,
                inp2_ptr,
                gamma_ptr,
                beta_ptr,
                epsilon,
                mean_ptr,
                rstd_ptr,
                batch_size,
                seq_len,
                channels,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        raise Error("Invalid target")


@compiler.register("layernorm_fused_residual_fwd")
struct LayerNormFusedResidualFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        residual: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        normed: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x1: InputTensor[dtype=dtype, rank=2, static_spec=...],
        x2: InputTensor[dtype=dtype, rank=2, static_spec=...],
        gamma: InputTensor[dtype=dtype, rank=1, static_spec=...],
        beta: InputTensor[dtype=dtype, rank=1, static_spec=...],
        epsilon: Scalar[DType.float32],
        mean: MutableInputTensor[dtype=DType.float32, rank=1, static_spec=...],
        rstd: MutableInputTensor[dtype=DType.float32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        channels: Int64,  # Our C
        ctx: DeviceContext,
    ) capturing raises:
        if residual.size() != Int(batch_size * seq_len * channels):
            raise Error("residual size mismatch")
        if normed.size() != Int(batch_size * seq_len * channels):
            raise Error("normed size mismatch")
        if x1.size() != Int(batch_size * seq_len * channels):
            raise Error("x1 size mismatch")
        if x2.size() != Int(batch_size * seq_len * channels):
            raise Error("x2 size mismatch")
        if gamma.size() != Int(channels):
            raise Error("gamma size mismatch")
        if beta.size() != Int(channels):
            raise Error("beta size mismatch")
        if mean.size() != Int(batch_size * seq_len):
            raise Error("mean size mismatch")
        if rstd.size() != Int(batch_size * seq_len):
            raise Error("rstd size mismatch")

        layernorm_fused_residual_fwd[dtype, target](
            residual.unsafe_ptr(),
            normed.unsafe_ptr(),
            x1.unsafe_ptr(),
            x2.unsafe_ptr(),
            gamma.unsafe_ptr(),
            beta.unsafe_ptr(),
            epsilon,
            mean.unsafe_ptr(),
            rstd.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# LayerNorm Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _layernorm_bwd_cpu[
    dtype: DType,
    width: Int,
](
    idx: Int,
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    dgamma_partial_ptr: MutKernelPtr[DType.float32],
    dbeta_partial_ptr: MutKernelPtr[DType.float32],
    channels: Int,
) -> None:
    var row_offset = idx * channels
    var mean = mean_ptr[idx]
    var rstd = rstd_ptr[idx]
    var mean_vec = SIMD[DType.float32, width](mean)
    var rstd_vec = SIMD[DType.float32, width](rstd)

    # Parameter gradients accumulate into this worker's private [channels]
    # partials (dgamma_partial_ptr/dbeta_partial_ptr index by channel only,
    # not by row). The caller owns one such pair per worker, so the row loop
    # races nothing and needs no atomics; the partials are summed into
    # d_gamma/d_beta once the workers join. This is a very similar approach to
    # the one used in the implementation of matmul. This is done in lue of using
    # atomics to accumulate the gradient of the weight and bias like in Karpathy's implementation.

    # Pass 1: Accumulate parameter gradients and compute row statistics.
    var sum_gdy_vec = SIMD[DType.float32, width](0.0)
    var sum_gdy_xhat_vec = SIMD[DType.float32, width](0.0)
    var i = 0
    while i + width <= channels:
        var dy = (
            (d_output_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var x = (
            (input_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var g = (gamma_ptr + i).load[width=width]().cast[DType.float32]()
        var x_hat = (x - mean_vec) * rstd_vec

        var dgamma_prev = (dgamma_partial_ptr + i).load[width=width]()
        var dbeta_prev = (dbeta_partial_ptr + i).load[width=width]()
        (dgamma_partial_ptr + i).store(fma(dy, x_hat, dgamma_prev))
        (dbeta_partial_ptr + i).store(dbeta_prev + dy)

        var gdy = g * dy
        sum_gdy_vec += gdy
        sum_gdy_xhat_vec += gdy * x_hat
        i += width

    var sum_gdy = sum_gdy_vec.reduce_add()
    var sum_gdy_xhat = sum_gdy_xhat_vec.reduce_add()
    for j in range(i, channels):
        var dy = d_output_ptr[row_offset + j].cast[DType.float32]()
        var x = input_ptr[row_offset + j].cast[DType.float32]()
        var g = gamma_ptr[j].cast[DType.float32]()
        var x_hat = (x - mean) * rstd

        dgamma_partial_ptr[j] += dy * x_hat
        dbeta_partial_ptr[j] += dy

        var gdy = g * dy
        sum_gdy += gdy
        sum_gdy_xhat += gdy * x_hat

    # Pass 2: Compute input gradient.
    var inv_c = 1.0 / Float32(channels)
    var s1 = sum_gdy * inv_c
    var s2 = sum_gdy_xhat * inv_c
    i = 0
    while i + width <= channels:
        var dy = (
            (d_output_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var g = (gamma_ptr + i).load[width=width]().cast[DType.float32]()
        var x = (
            (input_ptr + row_offset + i)
            .load[width=width]()
            .cast[DType.float32]()
        )
        var x_hat = (x - mean_vec) * rstd_vec
        var d_input = rstd_vec * (g * dy - s1 - (x_hat * s2))
        (d_input_ptr + row_offset + i).store(d_input.cast[dtype]())
        i += width

    for j in range(i, channels):
        var dy = d_output_ptr[row_offset + j].cast[DType.float32]()
        var g = gamma_ptr[j].cast[DType.float32]()
        var x = input_ptr[row_offset + j].cast[DType.float32]()
        var x_hat = (x - mean) * rstd
        var d_input = rstd * (g * dy - s1 - (x_hat * s2))
        d_input_ptr[row_offset + j] = d_input.cast[dtype]()


def layernorm_bwd_cpu[
    dtype: DType,
    width: Int,
    # Parameter-gradient dtype: production keeps pdtype == dtype; the test
    # harness (registered ops) accumulates d_gamma/d_beta in fp32.
    pdtype: DType = dtype,
](
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    d_gamma_ptr: MutKernelPtr[pdtype],
    d_beta_ptr: MutKernelPtr[pdtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) raises -> None:
    var total = Int(batch_size * seq_len)
    var c = Int(channels)
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(total, max_workers)
    var num_workers = ceildiv(total, rows_per_worker)

    # One private [channels] accumulator per worker for the parameter
    # gradients. Reducing these once after the join replaces the per-element
    # atomics: the same cross-row sum, but contention-free and vectorizable.
    var dgamma_partial = alloc[Scalar[DType.float32]](num_workers * c)
    var dbeta_partial = alloc[Scalar[DType.float32]](num_workers * c)

    @parameter
    def _worker(w: Int):
        var base = w * rows_per_worker
        var count = min(rows_per_worker, total - base)
        var dgamma_row = dgamma_partial + w * c
        var dbeta_row = dbeta_partial + w * c
        for j in range(c):
            dgamma_row[j] = 0.0
            dbeta_row[j] = 0.0
        for local in range(count):
            var idx = base + local
            _layernorm_bwd_cpu[dtype, width](
                idx,
                d_output_ptr,
                input_ptr,
                gamma_ptr,
                mean_ptr,
                rstd_ptr,
                d_input_ptr,
                rebind[MutKernelPtr[DType.float32]](
                    dgamma_row.as_unsafe_any_origin()
                ),
                rebind[MutKernelPtr[DType.float32]](
                    dbeta_row.as_unsafe_any_origin()
                ),
                c,
            )

    traced_parallelize["layernorm_bwd", _worker](num_workers)

    # Reduce the per-worker partials into the parameter gradients. We add into
    # whatever is already there (the atomic path accumulated the same way), so
    # callers that accumulate across micro-steps keep that behavior.
    for j in range(c):
        var acc_dgamma = Scalar[DType.float32](0.0)
        var acc_dbeta = Scalar[DType.float32](0.0)
        for w in range(num_workers):
            acc_dgamma += dgamma_partial[w * c + j]
            acc_dbeta += dbeta_partial[w * c + j]
        d_gamma_ptr[j] = (
            d_gamma_ptr[j].cast[DType.float32]() + acc_dgamma
        ).cast[pdtype]()
        d_beta_ptr[j] = (d_beta_ptr[j].cast[DType.float32]() + acc_dbeta).cast[
            pdtype
        ]()

    dgamma_partial.free()
    dbeta_partial.free()


@always_inline
def _layernorm_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    channels: Int,
) -> None:
    # d_input only: one block reduces a row's statistics with block.sum and
    # writes that row's (disjoint) d_input slice, so there is nothing to race.
    # The parameter gradients reduce over the opposite axis (all rows) and are
    # handled by _layernorm_dgamma_dbeta_gpu, which needs no atomics either.
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    # idx = row*channels + i, where i = tile_base + tid*width. tile_base is a
    # multiple of BLOCK_SPAN (= BLOCK_SIZE*width) and, when `aligned` is True,
    # channels is a multiple of width (the caller only sets aligned=True when
    # it has checked channels % width == 0 on the host), so idx is provably
    # width-aligned even though the compiler can't see it (row and tid are
    # runtime). Same fix as the fused classifier/adamw. When `aligned` is
    # False (odd channel counts, e.g. the equivalence-suite's 767), the
    # per-row base offset row*channels is not provably width-aligned, so we
    # fall back to a fully scalar (width=1) sweep that needs no alignment
    # guarantee at all.
    comptime align = align_of[SIMD[dtype, width]]()

    for row in range(block_row, num_rows, stride):
        var mean = mean_ptr[row]
        var rstd = rstd_ptr[row]
        var mean_vec = SIMD[DType.float32, width](mean)
        var rstd_vec = SIMD[DType.float32, width](rstd)
        var sum_gdy_thread = SIMD[DType.float32, width](0.0)
        var sum_gdy_xhat_thread = SIMD[DType.float32, width](0.0)
        var sum_gdy_tail = Scalar[DType.float32](0.0)
        var sum_gdy_xhat_tail = Scalar[DType.float32](0.0)

        # Pass 1: Row-statistic reduction (sum_gdy, sum_gdy_xhat).
        comptime if aligned:
            for tile_base in range(0, channels, BLOCK_SPAN):
                var i = tile_base + tid * width
                if i + width <= channels:
                    var idx = row * channels + i
                    var dy = (
                        (d_output_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x = (
                        (input_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var g = (
                        (gamma_ptr + i)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x_hat = (x - mean_vec) * rstd_vec

                    var gdy = g * dy
                    sum_gdy_thread += gdy
                    sum_gdy_xhat_thread += gdy * x_hat
                elif i < channels:
                    for j in range(i, channels):
                        var idx = row * channels + j
                        var dy = d_output_ptr[idx].cast[DType.float32]()
                        var x = input_ptr[idx].cast[DType.float32]()
                        var g = gamma_ptr[j].cast[DType.float32]()
                        var x_hat = (x - mean) * rstd

                        var gdy = g * dy
                        sum_gdy_tail += gdy
                        sum_gdy_xhat_tail += gdy * x_hat
        else:
            for i in range(tid, channels, BLOCK_SIZE):
                var idx = row * channels + i
                var dy = d_output_ptr[idx].cast[DType.float32]()
                var x = input_ptr[idx].cast[DType.float32]()
                var g = gamma_ptr[i].cast[DType.float32]()
                var x_hat = (x - mean) * rstd

                var gdy = g * dy
                sum_gdy_tail += gdy
                sum_gdy_xhat_tail += gdy * x_hat

        var sum_gdy = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_thread.reduce_add() + sum_gdy_tail
        )
        var sum_gdy_xhat = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_xhat_thread.reduce_add() + sum_gdy_xhat_tail
        )

        # Pass 2: Compute input gradient.
        var inv_c = 1.0 / Float32(channels)
        comptime if aligned:
            for tile_base in range(0, channels, BLOCK_SPAN):
                var i = tile_base + tid * width
                if i + width <= channels:
                    var idx = row * channels + i
                    var dy = (
                        (d_output_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x = (
                        (input_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var g = (
                        (gamma_ptr + i)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x_hat = (x - mean_vec) * rstd_vec
                    var d_input = rstd_vec * (
                        g * dy
                        - (sum_gdy * inv_c)
                        - (x_hat * sum_gdy_xhat * inv_c)
                    )
                    (d_input_ptr + idx).store[width=width, alignment=align](
                        d_input.cast[dtype]()
                    )
                elif i < channels:
                    for j in range(i, channels):
                        var idx = row * channels + j
                        var dy = d_output_ptr[idx].cast[DType.float32]()
                        var x = input_ptr[idx].cast[DType.float32]()
                        var g = gamma_ptr[j].cast[DType.float32]()
                        var x_hat = (x - mean) * rstd
                        var d_input = rstd * (
                            g * dy
                            - (sum_gdy * inv_c)
                            - (x_hat * sum_gdy_xhat * inv_c)
                        )
                        d_input_ptr[idx] = d_input.cast[dtype]()
        else:
            for i in range(tid, channels, BLOCK_SIZE):
                var idx = row * channels + i
                var dy = d_output_ptr[idx].cast[DType.float32]()
                var x = input_ptr[idx].cast[DType.float32]()
                var g = gamma_ptr[i].cast[DType.float32]()
                var x_hat = (x - mean) * rstd
                var d_input = rstd * (
                    g * dy - (sum_gdy * inv_c) - (x_hat * sum_gdy_xhat * inv_c)
                )
                d_input_ptr[idx] = d_input.cast[dtype]()


def layernorm_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    d_output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    _layernorm_bwd_gpu[dtype, BLOCK_SIZE, width, aligned=aligned](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        d_output_ptr,
        input_ptr,
        gamma_ptr,
        mean_ptr,
        rstd_ptr,
        d_input_ptr,
        Int(channels),
    )


@always_inline
def _ln_bwd_dparam_scratch(
    count: Int, ctx: DeviceContext
) raises -> MutKernelPtr[DType.float32]:
    # Persistent fp32 scratch for the fused LN-backward kernel's block-partial
    # dgamma/dbeta (see _layernorm_bwd_fused_gpu): dgamma partials live in
    # [0 : blocks_cap*CHANNELS_CAP), dbeta partials in
    # [blocks_cap*CHANNELS_CAP : 2*blocks_cap*CHANNELS_CAP). `count` is fixed
    # per process (SM_OVERPROVISION*num_sm is a per-device constant and
    # CHANNELS_CAP is a hardcoded bound — see the GPU dispatch in
    # layernorm_fused_residual_bwd), so — exactly like _ln_dparam_scratch
    # below — one allocation safely serves every call regardless of shape.
    # Never re-zeroed: each block writes its own [0:num_row_blocks) slot in
    # full every launch before the finalize reduction reads it, so stale
    # data outside the active grid (or from a prior launch) is never read.
    return persistent_device_buffer[DType.float32](
        ctx, "LN_BWD_DPARAM_SCRATCH", count
    )


def _layernorm_bwd_fused_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
    # When True, pass 2 additionally seeds the incoming residual-stream
    # gradient (resid_in_ptr) into d_inp1/d_inp2 in the SAME store as the LN
    # input-gradient accumulate, replacing the separate `residual_grad_
    # broadcast` kernel launch that used to run before this one. Ignored
    # (resid_in_ptr never read) when False — used for the ln_f call site,
    # which has no incoming residual gradient to seed.
    HAS_RESID_IN: Bool = False,
    # Per-block scratch stride / shared-memory bound for dgamma/dbeta.
    # 2048 covers every channel count this kernel is ever launched with
    # (GPT-2 configs top out at 1600; the equivalence suite tops out at 768)
    # — same bound already assumed by the forward fused kernel's LN_CAP.
    CHANNELS_CAP: Int = 2048,
](
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    resid_in_ptr: ImmutKernelPtr[dtype],
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    scratch_dgamma: MutKernelPtr[DType.float32],
    scratch_dbeta: MutKernelPtr[DType.float32],
    num_rows: Int,
    channels: Int,
    blocks_cap: Int,
) -> None:
    # Fused LN-backward MAIN pass (paired with the _ln_bwd_fused_finalize_gpu
    # launch below). One sweep over [B,T,C] produces d_input, folds the
    # residual-grad seed (HAS_RESID_IN), and reduces dgamma/dbeta.
    #
    # dgamma/dbeta accumulate into per-thread REGISTERS, not shared memory:
    # thread `tid` owns a fixed disjoint set of channel offsets on every row
    # (i = tile_base + tid*width aligned; i = col_base + tid scalar), so no
    # other thread touches its accumulator. The reduction is therefore
    # deterministic and needs only a single scratch flush at the end; the
    # cross-block sum is a separate finalize launch.
    #
    # `aligned` proof: the host dispatch sets aligned=True only when
    # channels % width == 0, making every row*channels+i offset width-aligned
    # and the scalar tail (i < channels < i+width) unreachable.
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    comptime align = align_of[SIMD[dtype, width]]()
    # Number of comptime-unrolled work chunks a single thread can own. For the
    # aligned (vector) path a chunk is `width` channels every BLOCK_SPAN; for
    # the scalar path a chunk is one channel every BLOCK_SIZE. Both are sized
    # to the CHANNELS_CAP bound so the register accumulators are a fixed,
    # small, comptime-known count (2 and 8 respectively at the defaults).
    comptime NUM_TILES = ceildiv(CHANNELS_CAP, BLOCK_SPAN)
    comptime NUM_COLS = ceildiv(CHANNELS_CAP, BLOCK_SIZE)

    var tid = Int(thread_idx.x)
    var block_row = Int(block_idx.x)
    var num_blocks = Int(grid_dim.x)
    var inv_c = 1.0 / Float32(channels)

    comptime if aligned:
        var dg_acc = InlineArray[SIMD[DType.float32, width], NUM_TILES](
            fill=SIMD[DType.float32, width](0.0)
        )
        var db_acc = InlineArray[SIMD[DType.float32, width], NUM_TILES](
            fill=SIMD[DType.float32, width](0.0)
        )

        for row in range(block_row, num_rows, num_blocks):
            var mean = mean_ptr[row]
            var rstd = rstd_ptr[row]
            var mean_vec = SIMD[DType.float32, width](mean)
            var rstd_vec = SIMD[DType.float32, width](rstd)
            var sum_gdy_thread = SIMD[DType.float32, width](0.0)
            var sum_gdy_xhat_thread = SIMD[DType.float32, width](0.0)

            # Pass 1: row-statistic reduction (sum_gdy, sum_gdy_xhat).
            comptime for t in range(NUM_TILES):
                comptime tile_base = t * BLOCK_SPAN
                var i = tile_base + tid * width
                if i + width <= channels:
                    var idx = row * channels + i
                    var dy = (
                        (d_output_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x = (
                        (input_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var g = (
                        (gamma_ptr + i)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x_hat = (x - mean_vec) * rstd_vec
                    var gdy = g * dy
                    sum_gdy_thread += gdy
                    sum_gdy_xhat_thread += gdy * x_hat

            var sum_gdy = block.sum[block_size=BLOCK_SIZE](
                sum_gdy_thread.reduce_add()
            )
            var sum_gdy_xhat = block.sum[block_size=BLOCK_SIZE](
                sum_gdy_xhat_thread.reduce_add()
            )

            # Pass 2: d_input, fused residual-grad seed (HAS_RESID_IN), and
            # dgamma/dbeta REGISTER accumulation — all in the same sweep.
            comptime for t in range(NUM_TILES):
                comptime tile_base = t * BLOCK_SPAN
                var i = tile_base + tid * width
                if i + width <= channels:
                    var idx = row * channels + i
                    var dy = (
                        (d_output_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x = (
                        (input_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var g = (
                        (gamma_ptr + i)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var x_hat = (x - mean_vec) * rstd_vec
                    var d_input = rstd_vec * (
                        g * dy
                        - (sum_gdy * inv_c)
                        - (x_hat * sum_gdy_xhat * inv_c)
                    )

                    dg_acc[t] += dy * x_hat
                    db_acc[t] += dy

                    var g1 = (
                        (d_inp1_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var g2 = (
                        (d_inp2_ptr + idx)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    comptime if HAS_RESID_IN:
                        var resid = (
                            (resid_in_ptr + idx)
                            .load[width=width, alignment=align]()
                            .cast[DType.float32]()
                        )
                        (d_inp1_ptr + idx).store[width=width, alignment=align](
                            (g1 + resid + d_input).cast[dtype]()
                        )
                        (d_inp2_ptr + idx).store[width=width, alignment=align](
                            (g2 + resid + d_input).cast[dtype]()
                        )
                    else:
                        (d_inp1_ptr + idx).store[width=width, alignment=align](
                            (g1 + d_input).cast[dtype]()
                        )
                        (d_inp2_ptr + idx).store[width=width, alignment=align](
                            (g2 + d_input).cast[dtype]()
                        )

        # Flush this thread's register partials to this block's slot in the
        # CHANNEL-MAJOR scratch (scratch[c * blocks_cap + block_row]). Laying
        # the partials out channel-major lets the finalize kernel reduce each
        # channel's blocks_cap partials with a single coalesced block (one
        # block per channel, saturating every SM) instead of the 3-block
        # thread-per-channel finalize a block-major layout would force (only
        # ceil(C/256)=3 blocks -> 45 idle SMs, which profiled at ~2.4 ms). The
        # flush is `width` strided scalar writes per owned chunk, done once per
        # block (not per row), so its cost is negligible.
        comptime for t in range(NUM_TILES):
            comptime tile_base = t * BLOCK_SPAN
            var i = tile_base + tid * width
            if i + width <= channels:
                comptime for lane in range(width):
                    var c = i + lane
                    scratch_dgamma[c * blocks_cap + block_row] = dg_acc[t][lane]
                    scratch_dbeta[c * blocks_cap + block_row] = db_acc[t][lane]
    else:
        var dg_acc = InlineArray[Scalar[DType.float32], NUM_COLS](
            fill=Scalar[DType.float32](0.0)
        )
        var db_acc = InlineArray[Scalar[DType.float32], NUM_COLS](
            fill=Scalar[DType.float32](0.0)
        )

        for row in range(block_row, num_rows, num_blocks):
            var mean = mean_ptr[row]
            var rstd = rstd_ptr[row]
            var sum_gdy_tail = Scalar[DType.float32](0.0)
            var sum_gdy_xhat_tail = Scalar[DType.float32](0.0)

            for i in range(tid, channels, BLOCK_SIZE):
                var idx = row * channels + i
                var dy = d_output_ptr[idx].cast[DType.float32]()
                var x = input_ptr[idx].cast[DType.float32]()
                var g = gamma_ptr[i].cast[DType.float32]()
                var x_hat = (x - mean) * rstd
                var gdy = g * dy
                sum_gdy_tail += gdy
                sum_gdy_xhat_tail += gdy * x_hat

            var sum_gdy = block.sum[block_size=BLOCK_SIZE](sum_gdy_tail)
            var sum_gdy_xhat = block.sum[block_size=BLOCK_SIZE](
                sum_gdy_xhat_tail
            )

            comptime for s in range(NUM_COLS):
                comptime col_base = s * BLOCK_SIZE
                var i = col_base + tid
                if i < channels:
                    var idx = row * channels + i
                    var dy = d_output_ptr[idx].cast[DType.float32]()
                    var x = input_ptr[idx].cast[DType.float32]()
                    var g = gamma_ptr[i].cast[DType.float32]()
                    var x_hat = (x - mean) * rstd
                    var d_input = rstd * (
                        g * dy
                        - (sum_gdy * inv_c)
                        - (x_hat * sum_gdy_xhat * inv_c)
                    )

                    dg_acc[s] += dy * x_hat
                    db_acc[s] += dy

                    var g1 = d_inp1_ptr[idx].cast[DType.float32]()
                    var g2 = d_inp2_ptr[idx].cast[DType.float32]()
                    comptime if HAS_RESID_IN:
                        var resid = resid_in_ptr[idx].cast[DType.float32]()
                        d_inp1_ptr[idx] = (g1 + resid + d_input).cast[dtype]()
                        d_inp2_ptr[idx] = (g2 + resid + d_input).cast[dtype]()
                    else:
                        d_inp1_ptr[idx] = (g1 + d_input).cast[dtype]()
                        d_inp2_ptr[idx] = (g2 + d_input).cast[dtype]()

        comptime for s in range(NUM_COLS):
            comptime col_base = s * BLOCK_SIZE
            var i = col_base + tid
            if i < channels:
                scratch_dgamma[i * blocks_cap + block_row] = dg_acc[s]
                scratch_dbeta[i * blocks_cap + block_row] = db_acc[s]


def _ln_bwd_fused_finalize_gpu[
    pdtype: DType,
    BLOCK_SIZE: Int,
](
    d_gamma_ptr: MutKernelPtr[pdtype],
    d_beta_ptr: MutKernelPtr[pdtype],
    scratch_dgamma: MutKernelPtr[DType.float32],
    scratch_dbeta: MutKernelPtr[DType.float32],
    num_blocks: Int,
    blocks_cap: Int,
) -> None:
    # Deterministic cross-block reduction of the per-block dgamma/dbeta
    # partials. ONE BLOCK PER CHANNEL (block_idx.x == channel) so all channels
    # reduce concurrently; each channel's blocks_cap partials are contiguous in
    # the channel-major scratch and read fully coalesced. Fixed block dims ->
    # fixed reduction tree -> bit-reproducible. d_gamma/d_beta ALWAYS
    # accumulate (grad accumulation across micro-steps). No re-zero needed: the
    # main kernel overwrites every [0:num_blocks) slot each launch.
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = col * blocks_cap
    var dg_thread = Scalar[DType.float32](0.0)
    var db_thread = Scalar[DType.float32](0.0)
    for b in range(tid, num_blocks, BLOCK_SIZE):
        dg_thread += scratch_dgamma[base + b]
        db_thread += scratch_dbeta[base + b]
    var dg = block.sum[block_size=BLOCK_SIZE](dg_thread)
    var db = block.sum[block_size=BLOCK_SIZE](db_thread)
    if tid == 0:
        d_gamma_ptr[col] = (d_gamma_ptr[col].cast[DType.float32]() + dg).cast[
            pdtype
        ]()
        d_beta_ptr[col] = (d_beta_ptr[col].cast[DType.float32]() + db).cast[
            pdtype
        ]()


@always_inline
def _ln_dparam_scratch(
    cap: Int, ctx: DeviceContext
) raises -> MutKernelPtr[DType.float32]:
    # Persistent fp32 scratch holding dgamma partials in [0:cap] and dbeta
    # partials in [cap:2*cap]. Allocate-once (heap-held via a device-keyed
    # process global), zeroed; the finalize kernel re-zeros after each use.
    return persistent_device_buffer[DType.float32](
        ctx, "LN_DPARAM_SCRATCH", 2 * cap, zero=True
    )


def _ln_dparam_accum_gpu[
    dtype: DType,
](
    scratch: MutKernelPtr[DType.float32],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    num_rows: Int,
    channels: Int,
    row_tile: Int,
    cap: Int,
) -> None:
    # dgamma[c] += sum_r dy[r,c]·x_hat[r,c]; dbeta[c] += sum_r dy[r,c]. One thread
    # per channel (adjacent threads → adjacent columns → COALESCED reads of
    # d_output/input), grid.y row-blocks for occupancy; per-block partials are
    # atomically accumulated into the fp32 scratch (dgamma in [0:cap], dbeta in
    # [cap:]). Replaces the uncoalesced row-strided block.sum reduction.
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if col >= channels:
        return
    var r0 = Int(block_idx.y) * row_tile
    var r1 = min(r0 + row_tile, num_rows)
    var acc_dg = Scalar[DType.float32](0.0)
    var acc_db = Scalar[DType.float32](0.0)
    for r in range(r0, r1):
        var off = r * channels + col
        var dy = d_output_ptr[off].cast[DType.float32]()
        var x = input_ptr[off].cast[DType.float32]()
        var x_hat = (x - mean_ptr[r]) * rstd_ptr[r]
        acc_dg = fma(dy, x_hat, acc_dg)
        acc_db += dy
    _ = Atomic[DType.float32].fetch_add(scratch + col, acc_dg)
    _ = Atomic[DType.float32].fetch_add(scratch + cap + col, acc_db)


def _ln_dparam_finalize_gpu[
    dtype: DType,
](
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
    scratch: MutKernelPtr[DType.float32],
    channels: Int,
    cap: Int,
) -> None:
    # d_gamma/d_beta ALWAYS accumulate (grad accumulation across micro-steps).
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if col < channels:
        var dg = scratch[col]
        var db = scratch[cap + col]
        scratch[col] = 0.0
        scratch[cap + col] = 0.0
        d_gamma_ptr[col] = (d_gamma_ptr[col].cast[DType.float32]() + dg).cast[
            dtype
        ]()
        d_beta_ptr[col] = (d_beta_ptr[col].cast[DType.float32]() + db).cast[
            dtype
        ]()


def _layernorm_dgamma_dbeta_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
](
    num_rows: Int,
    channels: Int,
    tid: Int,
    stride: Int,
    block_col: Int,
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
) -> None:
    # Channel-parallel parameter gradients (mirrors matmul_bias_bwd_gpu): each
    # block owns channel tiles and reduces over all rows with block.sum, so a
    # single thread writes each channel. No atomics, no cross-block sharing.
    # d_gamma[c] = sum_r dy[r, c] * x_hat[r, c]; d_beta[c] = sum_r dy[r, c].
    var num_tiles = ceildiv(channels, width)
    for tile in range(block_col, num_tiles, stride):
        var base = tile * width
        if base + width <= channels:
            var acc_dgamma = SIMD[DType.float32, width](0.0)
            var acc_dbeta = SIMD[DType.float32, width](0.0)
            for r in range(tid, num_rows, BLOCK_SIZE):
                var off = r * channels + base
                var dy = (
                    (d_output_ptr + off)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var x = (
                    (input_ptr + off).load[width=width]().cast[DType.float32]()
                )
                var x_hat = (x - mean_ptr[r]) * rstd_ptr[r]
                acc_dgamma = fma(dy, x_hat, acc_dgamma)
                acc_dbeta += dy
            var sum_dgamma = SIMD[DType.float32, width](0.0)
            var sum_dbeta = SIMD[DType.float32, width](0.0)
            comptime for i in range(width):
                sum_dgamma[i] = block.sum[block_size=BLOCK_SIZE](acc_dgamma[i])
                sum_dbeta[i] = block.sum[block_size=BLOCK_SIZE](acc_dbeta[i])
            if tid == 0:
                # Accumulate into existing grads (matches the CPU path and the
                # old atomic accumulate-into-buffer behavior).
                var prev_dgamma = (
                    (d_gamma_ptr + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var prev_dbeta = (
                    (d_beta_ptr + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                (d_gamma_ptr + base).store(
                    (prev_dgamma + sum_dgamma).cast[dtype]()
                )
                (d_beta_ptr + base).store(
                    (prev_dbeta + sum_dbeta).cast[dtype]()
                )
        else:
            # Ragged tail. Loop bounds depend only on `base`, so every thread
            # reaches block.sum uniformly.
            for c in range(base, channels):
                var acc_dgamma = Scalar[DType.float32](0.0)
                var acc_dbeta = Scalar[DType.float32](0.0)
                for r in range(tid, num_rows, BLOCK_SIZE):
                    var off = r * channels + c
                    var dy = d_output_ptr[off].cast[DType.float32]()
                    var x = input_ptr[off].cast[DType.float32]()
                    var x_hat = (x - mean_ptr[r]) * rstd_ptr[r]
                    acc_dgamma += dy * x_hat
                    acc_dbeta += dy
                var sum_dgamma = block.sum[block_size=BLOCK_SIZE](acc_dgamma)
                var sum_dbeta = block.sum[block_size=BLOCK_SIZE](acc_dbeta)
                if tid == 0:
                    d_gamma_ptr[c] = (
                        d_gamma_ptr[c].cast[DType.float32]() + sum_dgamma
                    ).cast[dtype]()
                    d_beta_ptr[c] = (
                        d_beta_ptr[c].cast[DType.float32]() + sum_dbeta
                    ).cast[dtype]()


def layernorm_dgamma_dbeta_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
](
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    _layernorm_dgamma_dbeta_gpu[dtype, BLOCK_SIZE, width](
        Int(batch_size * seq_len),
        Int(channels),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        d_output_ptr,
        input_ptr,
        mean_ptr,
        rstd_ptr,
        d_gamma_ptr,
        d_beta_ptr,
    )


@always_inline
def _layernorm_dparam_gpu[
    dtype: DType,
    # Parameter-gradient dtype: production keeps pdtype == dtype; the test
    # harness (registered ops) accumulates d_gamma/d_beta in fp32.
    pdtype: DType = dtype,
](
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_gamma_ptr: MutKernelPtr[pdtype],
    d_beta_ptr: MutKernelPtr[pdtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    # d_gamma/d_beta as a coalesced, row-parallel reduction (one thread per
    # channel, grid.y row-blocks, fp32 atomics into scratch) + a finalize
    # pass — mirrors the ⑰ dbias fix, replacing the uncoalesced row-strided
    # block.sum reduction. Shared by both layernorm_bwd and
    # layernorm_fused_residual_bwd's GPU paths (identical dgamma/dbeta work
    # in both cases; only the d_input kernel differs).
    comptime BLOCK_SIZE = 256
    comptime LN_ROW_BLOCKS = 16
    var device_ctx = ctx
    var ln_rows = Int(batch_size * seq_len)
    var ln_ch = Int(channels)
    var ln_row_tile = ceildiv(ln_rows, LN_ROW_BLOCKS)
    var ln_col_blocks = max(ceildiv(ln_ch, BLOCK_SIZE), 1)
    var ln_scratch = _ln_dparam_scratch(65536, device_ctx)

    comptime ln_accum_k = _ln_dparam_accum_gpu[dtype]
    var ln_accum_c = device_ctx.compile_function[ln_accum_k]()
    device_ctx.enqueue_function(
        ln_accum_c,
        ln_scratch,
        d_output_ptr,
        input_ptr,
        mean_ptr,
        rstd_ptr,
        ln_rows,
        ln_ch,
        ln_row_tile,
        65536,
        grid_dim=(ln_col_blocks, LN_ROW_BLOCKS),
        block_dim=(BLOCK_SIZE,),
    )
    comptime ln_fin_k = _ln_dparam_finalize_gpu[pdtype]
    var ln_fin_c = device_ctx.compile_function[ln_fin_k]()
    device_ctx.enqueue_function(
        ln_fin_c,
        d_gamma_ptr,
        d_beta_ptr,
        ln_scratch,
        ln_ch,
        65536,
        grid_dim=(ln_col_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def layernorm_bwd[
    dtype: DType,
    target: StaticString,
    # Parameter-gradient dtype: production keeps pdtype == dtype; the test
    # harness (registered ops) accumulates d_gamma/d_beta in fp32.
    pdtype: DType = dtype,
](
    d_output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    d_gamma_ptr: MutKernelPtr[pdtype],
    d_beta_ptr: MutKernelPtr[pdtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        layernorm_bwd_cpu[dtype, width, pdtype](
            d_output_ptr,
            input_ptr,
            gamma_ptr,
            mean_ptr,
            rstd_ptr,
            d_input_ptr,
            d_gamma_ptr,
            d_beta_ptr,
            batch_size,
            seq_len,
            channels,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        comptime width = 4
        # NOTE: This two kernel method needs to be profiled on the GPU.
        # Karpathy uses a single kernel to compute the gradient of the input, weight, and bias.
        # In theory this should be faster than the two kernel method, but mojo's compiler is very smart
        # and it might be able to optimize the two kernel method to be as fast as the single kernel method.
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        # Kernel 1: d_input, one block per row tile (reduces over channels).
        # Dispatch aligned vs. scalar-fallback kernels at the host: the
        # vectorized fast path requires channels % width == 0 (so every
        # row's base offset row*channels is width-aligned); production
        # shapes satisfy this, but the equivalence suite's odd channel
        # counts (e.g. 767) don't, and would otherwise crash with
        # CUDA_ERROR_MISALIGNED_ADDRESS. See _layernorm_bwd_gpu.
        var num_row_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        if Int(channels) % width == 0:
            comptime dinput_kernel = layernorm_bwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=True
            ]
            var dinput_compiled = device_ctx.compile_function[dinput_kernel]()
            device_ctx.enqueue_function(
                dinput_compiled,
                d_output_ptr,
                input_ptr,
                gamma_ptr,
                mean_ptr,
                rstd_ptr,
                d_input_ptr,
                batch_size,
                seq_len,
                channels,
                grid_dim=(num_row_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            comptime dinput_kernel_u = layernorm_bwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=False
            ]
            var dinput_compiled_u = device_ctx.compile_function[
                dinput_kernel_u
            ]()
            device_ctx.enqueue_function(
                dinput_compiled_u,
                d_output_ptr,
                input_ptr,
                gamma_ptr,
                mean_ptr,
                rstd_ptr,
                d_input_ptr,
                batch_size,
                seq_len,
                channels,
                grid_dim=(num_row_blocks,),
                block_dim=(BLOCK_SIZE,),
            )

        # Kernel 2: d_gamma/d_beta.
        _layernorm_dparam_gpu[dtype, pdtype](
            d_output_ptr,
            input_ptr,
            mean_ptr,
            rstd_ptr,
            d_gamma_ptr,
            d_beta_ptr,
            batch_size,
            seq_len,
            channels,
            device_ctx,
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# LayerNorm Fused Residual Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _layernorm_fused_residual_bwd_broadcast_tile[
    dtype: DType,
    width: Int,
    aligned: Bool = False,
](
    idx: Int,
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    d_residual_ptr: ImmutKernelPtr[dtype],
) -> None:
    # `aligned` is opt-in (default False so the CPU vectorize() call site,
    # whose chunk offsets aren't provably width-aligned, is untouched). The
    # GPU dispatch below passes aligned=True: idx = global_tid*width there is
    # provably width-aligned (same proof as adamw's idx).
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        var grad = (d_residual_ptr + idx).load[width=width, alignment=align]()
        (d_inp1_ptr + idx).store[width=width, alignment=align](
            (d_inp1_ptr + idx).load[width=width, alignment=align]() + grad
        )
        (d_inp2_ptr + idx).store[width=width, alignment=align](
            (d_inp2_ptr + idx).load[width=width, alignment=align]() + grad
        )
    else:
        var grad = (d_residual_ptr + idx).load[width=width]()
        (d_inp1_ptr + idx).store((d_inp1_ptr + idx).load[width=width]() + grad)
        (d_inp2_ptr + idx).store((d_inp2_ptr + idx).load[width=width]() + grad)


@always_inline
def _layernorm_fused_residual_bwd_broadcast_cpu[
    dtype: DType,
    width: Int,
](
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    d_residual_ptr: ImmutKernelPtr[dtype],
    count: Int,
) -> None:
    @always_inline
    def _simd[w: Int](local: Int) {d_inp1_ptr, d_inp2_ptr, d_residual_ptr}:
        _layernorm_fused_residual_bwd_broadcast_tile[dtype, w](
            local, d_inp1_ptr, d_inp2_ptr, d_residual_ptr
        )

    vectorize[width, unroll_factor=UNROLL](count, _simd)


def layernorm_fused_residual_bwd_broadcast_gpu[
    dtype: DType,
    width: Int = 4,
](
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    d_residual_ptr: ImmutKernelPtr[dtype],
    count: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= count:
        _layernorm_fused_residual_bwd_broadcast_tile[
            dtype, width, aligned=True
        ](idx, d_inp1_ptr, d_inp2_ptr, d_residual_ptr)
    elif idx < count:
        for i in range(idx, count):
            _layernorm_fused_residual_bwd_broadcast_tile[dtype, 1](
                i, d_inp1_ptr, d_inp2_ptr, d_residual_ptr
            )


def residual_grad_broadcast[
    dtype: DType,
    target: StaticString,
](
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    count: Int,
    ctx: DeviceContext,
) capturing raises:
    """Residual-skip gradient carry: d_inp1 += src; d_inp2 += src.

    The fused residual layernorm backward computes only the layernorm
    input-gradient and accumulates it into d_inp1/d_inp2. The forward op is
    `out = LayerNorm(inp1 + inp2)`, so the true input gradients are
    `LN_dinp + d(inp1+inp2)` where `d(inp1+inp2)` is the incoming residual-
    stream gradient. This helper seeds that incoming residual gradient into
    d_inp1/d_inp2 before the fused backward runs, so the residual identity
    skip is preserved (without it the block gradients decay geometrically
    with depth). Dispatches CPU/GPU like the other kernels here."""
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        _layernorm_fused_residual_bwd_broadcast_cpu[dtype, width](
            d_inp1_ptr, d_inp2_ptr, src_ptr, count
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime width = 4
        var device_ctx = ctx
        var num_threads = ceildiv(count, width)
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)
        comptime gpu_kernel = layernorm_fused_residual_bwd_broadcast_gpu[
            dtype, width
        ]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            d_inp1_ptr,
            d_inp2_ptr,
            src_ptr,
            count,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


def layernorm_fused_residual_bwd[
    dtype: DType,
    target: StaticString,
    # Parameter-gradient dtype: production keeps pdtype == dtype; the test
    # harness (registered ops) accumulates d_gamma/d_beta in fp32.
    pdtype: DType = dtype,
    # When True, the incoming residual-stream gradient at resid_in_ptr is
    # fused into the same d_inp1/d_inp2 accumulate that the LN input-
    # gradient uses (GPU: inside the single fused kernel; CPU: a broadcast
    # pass before layernorm_bwd, matching the old call-site order), instead
    # of the caller running a separate `residual_grad_broadcast` launch
    # first. False (the default, used by e.g. the ln_f call site and the
    # registered custom op) means there is no incoming residual gradient to
    # seed and resid_in_ptr is ignored.
    HAS_RESID_IN: Bool = False,
](
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    d_output_ptr: MutKernelPtr[dtype],
    residual_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_gamma_ptr: MutKernelPtr[pdtype],
    d_beta_ptr: MutKernelPtr[pdtype],
    d_residual_ptr: MutKernelPtr[dtype],
    resid_in_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        # CPU: seed broadcast (if HAS_RESID_IN) then layernorm_bwd then
        # broadcast. The seed call lives here (not at the train_gpt2.mojo
        # call site) so both targets share one signature; arithmetic and
        # kernel order are unchanged.
        comptime if HAS_RESID_IN:
            _layernorm_fused_residual_bwd_broadcast_cpu[dtype, width](
                d_inp1_ptr,
                d_inp2_ptr,
                resid_in_ptr,
                Int(batch_size * seq_len * channels),
            )
        layernorm_bwd[dtype, target, pdtype](
            d_output_ptr,
            residual_ptr,
            gamma_ptr,
            mean_ptr,
            rstd_ptr,
            d_residual_ptr,
            d_gamma_ptr,
            d_beta_ptr,
            batch_size,
            seq_len,
            channels,
            ctx,
        )
        _layernorm_fused_residual_bwd_broadcast_cpu[dtype, width](
            d_inp1_ptr,
            d_inp2_ptr,
            d_residual_ptr,
            Int(batch_size * seq_len * channels),
        )
    elif is_gpu[target]():
        # GPU: a 2-kernel fused LN backward. The MAIN kernel
        # (_layernorm_bwd_fused_gpu) computes d_input, folds the resid_in seed
        # (HAS_RESID_IN), and reduces dgamma/dbeta into per-block scratch
        # partials in one sweep; a tiny, fully parallel finalize kernel
        # (_ln_bwd_fused_finalize_gpu) then sums those partials into
        # d_gamma/d_beta. Together they replace what used to be up to 3 kernels
        # here (the d_input+broadcast kernel plus the dparam accum/finalize
        # pair) and, at each train_gpt2.mojo call site, a 4th kernel (the
        # separate `residual_grad_broadcast` launch).
        comptime BLOCK_SIZE = 256
        # SM_OVERPROVISION=3: at 72 reg/thread the kernel is register-bound to
        # 3 blocks/SM, so 3*num_sm saturates every SM in one wave. Fewer
        # blocks means fewer per-block partials, shrinking both the scratch
        # flush and the finalize reduction.
        comptime SM_OVERPROVISION = 3
        comptime width = 4
        comptime CHANNELS_CAP = 2048
        comptime FINALIZE_BLOCK = 256
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var ch = Int(channels)
        if ch > CHANNELS_CAP:
            raise Error(
                "layernorm_fused_residual_bwd: channels exceeds the fused"
                " backward kernel's CHANNELS_CAP"
            )
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        # SM_OVERPROVISION*num_sm is a per-device constant (num_sm never
        # changes within a process), so it upper-bounds num_row_blocks for
        # every shape this process ever launches with — the scratch buffer
        # below is sized once from this bound and is safe to reuse forever.
        var blocks_cap = SM_OVERPROVISION * num_sm
        var num_row_blocks = max(min(num_rows, blocks_cap), 1)

        var dparam_scratch = _ln_bwd_dparam_scratch(
            2 * blocks_cap * CHANNELS_CAP, device_ctx
        )
        var scratch_dgamma = dparam_scratch
        var scratch_dbeta = dparam_scratch + blocks_cap * CHANNELS_CAP

        # Pass 1 (main): d_input + resid seed + per-block dgamma/dbeta partials.
        # Same aligned/scalar-fallback dispatch as the kernels this replaces;
        # see _layernorm_bwd_fused_gpu's docstring.
        if ch % width == 0:
            comptime fused_k = _layernorm_bwd_fused_gpu[
                dtype,
                BLOCK_SIZE,
                width,
                aligned=True,
                HAS_RESID_IN=HAS_RESID_IN,
                CHANNELS_CAP=CHANNELS_CAP,
            ]
            var compiled = device_ctx.compile_function[fused_k]()
            device_ctx.enqueue_function(
                compiled,
                d_output_ptr,
                residual_ptr,
                gamma_ptr,
                mean_ptr,
                rstd_ptr,
                resid_in_ptr,
                d_inp1_ptr,
                d_inp2_ptr,
                scratch_dgamma,
                scratch_dbeta,
                num_rows,
                ch,
                blocks_cap,
                grid_dim=(num_row_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            comptime fused_k_u = _layernorm_bwd_fused_gpu[
                dtype,
                BLOCK_SIZE,
                width,
                aligned=False,
                HAS_RESID_IN=HAS_RESID_IN,
                CHANNELS_CAP=CHANNELS_CAP,
            ]
            var compiled_u = device_ctx.compile_function[fused_k_u]()
            device_ctx.enqueue_function(
                compiled_u,
                d_output_ptr,
                residual_ptr,
                gamma_ptr,
                mean_ptr,
                rstd_ptr,
                resid_in_ptr,
                d_inp1_ptr,
                d_inp2_ptr,
                scratch_dgamma,
                scratch_dbeta,
                num_rows,
                ch,
                blocks_cap,
                grid_dim=(num_row_blocks,),
                block_dim=(BLOCK_SIZE,),
            )

        # Pass 2 (finalize): deterministic parallel reduction of the
        # num_row_blocks per-block partials into d_gamma/d_beta — ONE BLOCK PER
        # CHANNEL (grid.x == channels) so every SM stays busy, reading each
        # channel's contiguous partials coalesced.
        comptime finalize_k = _ln_bwd_fused_finalize_gpu[pdtype, FINALIZE_BLOCK]
        var fcompiled = device_ctx.compile_function[finalize_k]()
        device_ctx.enqueue_function(
            fcompiled,
            d_gamma_ptr,
            d_beta_ptr,
            scratch_dgamma,
            scratch_dbeta,
            num_row_blocks,
            blocks_cap,
            grid_dim=(max(ch, 1),),
            block_dim=(FINALIZE_BLOCK,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("layernorm_bwd")
struct LayerNormBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_output: InputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        gamma: InputTensor[dtype=dtype, rank=1, static_spec=...],
        mean: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        rstd: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        d_x: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        # The pytest harness accumulates parameter grads in fp32 for every
        # kernel dtype (see tests/kernels/layernorm.py), so the registered op
        # takes fp32 here and forwards pdtype=float32. Production training
        # calls layernorm_bwd directly with pdtype == dtype.
        d_gamma: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_beta: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        channels: Int64,  # Our C
        ctx: DeviceContext,
    ) capturing raises:
        if d_output.size() != Int(batch_size * seq_len * channels):
            raise Error("d_output size mismatch")
        if x.size() != Int(batch_size * seq_len * channels):
            raise Error("x size mismatch")
        if gamma.size() != Int(channels):
            raise Error("gamma size mismatch")
        if mean.size() != Int(batch_size * seq_len):
            raise Error("mean size mismatch")
        if rstd.size() != Int(batch_size * seq_len):
            raise Error("rstd size mismatch")
        if d_x.size() != Int(batch_size * seq_len * channels):
            raise Error("d_x size mismatch")
        if d_gamma.size() != Int(channels):
            raise Error("d_gamma size mismatch")
        if d_beta.size() != Int(channels):
            raise Error("d_beta size mismatch")

        layernorm_bwd[dtype, target, DType.float32](
            d_output.unsafe_ptr(),
            x.unsafe_ptr(),
            gamma.unsafe_ptr(),
            mean.unsafe_ptr(),
            rstd.unsafe_ptr(),
            d_x.unsafe_ptr(),
            d_gamma.unsafe_ptr(),
            d_beta.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            ctx,
        )


@compiler.register("layernorm_fused_residual_bwd")
struct LayerNormFusedResidualBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_inp1: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_inp2: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_output: InputTensor[dtype=dtype, rank=2, static_spec=...],
        residual: InputTensor[dtype=dtype, rank=2, static_spec=...],
        gamma: InputTensor[dtype=dtype, rank=1, static_spec=...],
        mean: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        rstd: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        # fp32 parameter grads for the pytest harness; see LayerNormBwd.
        d_gamma: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_beta: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_residual: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        channels: Int64,  # Our C
        ctx: DeviceContext,
    ) capturing raises:
        var row_elems = Int(batch_size * seq_len * channels)
        if d_inp1.size() != row_elems:
            raise Error("d_inp1 size mismatch")
        if d_inp2.size() != row_elems:
            raise Error("d_inp2 size mismatch")
        if d_output.size() != row_elems:
            raise Error("d_output size mismatch")
        if residual.size() != row_elems:
            raise Error("residual size mismatch")
        if gamma.size() != Int(channels):
            raise Error("gamma size mismatch")
        if mean.size() != Int(batch_size * seq_len):
            raise Error("mean size mismatch")
        if rstd.size() != Int(batch_size * seq_len):
            raise Error("rstd size mismatch")
        if d_gamma.size() != Int(channels):
            raise Error("d_gamma size mismatch")
        if d_beta.size() != Int(channels):
            raise Error("d_beta size mismatch")
        if d_residual.size() != row_elems:
            raise Error("d_residual size mismatch")

        layernorm_fused_residual_bwd[dtype, target, DType.float32](
            d_inp1.unsafe_ptr(),
            d_inp2.unsafe_ptr(),
            d_output.unsafe_ptr(),
            residual.unsafe_ptr(),
            gamma.unsafe_ptr(),
            mean.unsafe_ptr(),
            rstd.unsafe_ptr(),
            d_gamma.unsafe_ptr(),
            d_beta.unsafe_ptr(),
            d_residual.unsafe_ptr(),
            # resid_in_ptr: unused placeholder — HAS_RESID_IN defaults False
            # for this registered op, so the fused kernel never reads it
            # (matches the pre-fusion tested "+=" accumulate contract; see
            # tests/test_layernorm_equivalence.py's *_accumulates tests).
            d_residual.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            ctx,
        )
