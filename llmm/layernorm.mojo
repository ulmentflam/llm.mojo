import compiler
from std.memory import alloc
from std.sys import simd_width_of, align_of
from std.gpu.primitives import block, warp
from std.gpu import WARP_SIZE
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
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
from std.ffi import _get_global_or_null, external_call
from layout import Layout, LayoutTensor

from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr

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

    # Handle the tail of the mean.
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

    # Handle the tail of the variance.
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

    # Handle the tail of the output.
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

        # Store the Mean and RSTD
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

    # Handle the tail of the mean.
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

    # Handle the tail of the variance.
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

    # Handle the tail of the output.
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

    for row in range(block_row, num_rows, stride):
        # Pass 1: Add input tensors, store to residual (global + shared), mean.
        # Explicit 16-byte alignment so the wide loads/stores are emitted (idx is
        # naturally aligned but the compiler can't prove it — Vp/channels runtime).
        comptime align = align_of[SIMD[dtype, width]]()
        var sum_thread = SIMD[DType.float32, width](0.0)
        var sum_tail = Scalar[DType.float32](0.0)

        for tile_base in range(0, channels, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= channels:
                var idx = row * channels + lane_base
                var val1 = (inp1_ptr + idx).load[width=width, alignment=align]()
                var val2 = (inp2_ptr + idx).load[width=width, alignment=align]()
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
        barrier()

        var mean_thread = sum_thread.reduce_add() + sum_tail
        var mean = block.sum[block_size=BLOCK_SIZE](mean_thread) / Float32(
            channels
        )

        # Pass 2: Variance (reads residual from shared).
        var var_thread = SIMD[DType.float32, width](0.0)
        var var_tail = Scalar[DType.float32](0.0)
        var mean_vec = SIMD[DType.float32, width](mean)
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

        var variance_thread = var_thread.reduce_add() + var_tail
        var variance = block.sum[block_size=BLOCK_SIZE](
            variance_thread
        ) / Float32(channels)
        var rstd = rsqrt(variance + epsilon)

        # Pass 3: Output (Normed) — read residual from shared.
        var rstd_vec = SIMD[DType.float32, width](rstd)
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

        # Store the Mean and RSTD
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
    _layernorm_fused_residual_fwd_gpu[dtype, BLOCK_SIZE, width](
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
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        comptime gpu_kernel = layernorm_fused_residual_fwd_gpu[
            dtype, BLOCK_SIZE
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
](
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
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
        ).cast[dtype]()
        d_beta_ptr[j] = (d_beta_ptr[j].cast[DType.float32]() + acc_dbeta).cast[
            dtype
        ]()

    dgamma_partial.free()
    dbeta_partial.free()


@always_inline
def _layernorm_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
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
    # multiple of BLOCK_SPAN (= BLOCK_SIZE*width) and channels (the model's C,
    # e.g. 768/1024/1280/1600) is always a multiple of width, so idx is
    # provably width-aligned even though the compiler can't see it (row and
    # tid are runtime). Same fix as the fused classifier/adamw.
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

        var sum_gdy = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_thread.reduce_add() + sum_gdy_tail
        )
        var sum_gdy_xhat = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_xhat_thread.reduce_add() + sum_gdy_xhat_tail
        )

        # Pass 2: Compute input gradient.
        var inv_c = 1.0 / Float32(channels)
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
                    g * dy - (sum_gdy * inv_c) - (x_hat * sum_gdy_xhat * inv_c)
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


def layernorm_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
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
    _layernorm_bwd_gpu[dtype, BLOCK_SIZE, width](
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
def _layernorm_bwd_residual_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    store_scratch: Bool = True,
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
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    channels: Int,
) -> None:
    # Fused variant of _layernorm_bwd_gpu for layernorm_fused_residual_bwd:
    # pass 2 computes dval in registers exactly as the plain kernel does, and
    # additionally does the broadcast accumulate inline
    # (d_inp1[idx] += dval; d_inp2[idx] += dval), replacing the separate
    # layernorm_fused_residual_bwd_broadcast_gpu launch + its d_residual
    # read. This is bit-identical to the two-kernel sequence: dval is
    # unchanged arithmetic, and the RMW matches
    # _layernorm_fused_residual_bwd_broadcast_tile's aligned=True path.
    #
    # store_scratch (comptime) additionally writes dval to d_input_ptr
    # (the scratch plane), matching the original kernel's store. Stage 1
    # keeps this (bit-identical by construction, since the fused broadcast
    # no longer reads it back). Stage 2 drops it: the scratch plane's
    # post-broadcast value is dead within this step (every access after the
    # broadcast read is a write) and is re-zeroed before next step's use, so
    # skipping the store changes no observable value, only saves the write.
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    # Same alignment proof as _layernorm_bwd_gpu: idx = row*channels + i is
    # provably width-aligned (channels % width == 0, tile_base is a multiple
    # of BLOCK_SPAN).
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

        var sum_gdy = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_thread.reduce_add() + sum_gdy_tail
        )
        var sum_gdy_xhat = block.sum[block_size=BLOCK_SIZE](
            sum_gdy_xhat_thread.reduce_add() + sum_gdy_xhat_tail
        )

        # Pass 2: Compute input gradient, then fuse the broadcast accumulate.
        var inv_c = 1.0 / Float32(channels)
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
                    g * dy - (sum_gdy * inv_c) - (x_hat * sum_gdy_xhat * inv_c)
                )
                comptime if store_scratch:
                    (d_input_ptr + idx).store[width=width, alignment=align](
                        d_input.cast[dtype]()
                    )
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
                (d_inp1_ptr + idx).store[width=width, alignment=align](
                    (g1 + d_input).cast[dtype]()
                )
                (d_inp2_ptr + idx).store[width=width, alignment=align](
                    (g2 + d_input).cast[dtype]()
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
                    comptime if store_scratch:
                        d_input_ptr[idx] = d_input.cast[dtype]()
                    var g1 = d_inp1_ptr[idx].cast[DType.float32]()
                    var g2 = d_inp2_ptr[idx].cast[DType.float32]()
                    d_inp1_ptr[idx] = (g1 + d_input).cast[dtype]()
                    d_inp2_ptr[idx] = (g2 + d_input).cast[dtype]()


def layernorm_bwd_residual_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    store_scratch: Bool = True,
](
    d_output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    _layernorm_bwd_residual_gpu[dtype, BLOCK_SIZE, width, store_scratch](
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
        d_inp1_ptr,
        d_inp2_ptr,
        Int(channels),
    )


@always_inline
def _ln_dparam_scratch(
    cap: Int, ctx: DeviceContext
) raises -> MutKernelPtr[DType.float32]:
    # Persistent fp32 scratch holding dgamma partials in [0:cap] and dbeta
    # partials in [cap:2*cap]. Allocate-once (heap-held via a device-keyed
    # process global), zeroed; the finalize kernel re-zeros after each use.
    comptime BufType = type_of(ctx.enqueue_create_buffer[DType.float32](1))
    var name = String(t"LLMM_LN_DPARAM_SCRATCH_{ctx.id()}")
    if gp := _get_global_or_null(name):
        var p = gp.value().bitcast[BufType]()
        return rebind[MutKernelPtr[DType.float32]](p[].unsafe_ptr())
    var buf = ctx.enqueue_create_buffer[DType.float32](2 * cap)
    ctx.enqueue_memset(buf, Float32(0))
    var hp = alloc[BufType](1)
    hp.init_pointee_move(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return rebind[MutKernelPtr[DType.float32]](hp[].unsafe_ptr())


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
    comptime ln_fin_k = _ln_dparam_finalize_gpu[dtype]
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
](
    d_output_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_input_ptr: MutKernelPtr[dtype],
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        layernorm_bwd_cpu[dtype, width](
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
        var num_row_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        comptime dinput_kernel = layernorm_bwd_gpu[dtype, BLOCK_SIZE, width]
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

        # Kernel 2: d_gamma/d_beta.
        _layernorm_dparam_gpu[dtype](
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


def layernorm_fused_residual_bwd[
    dtype: DType,
    target: StaticString,
](
    d_inp1_ptr: MutKernelPtr[dtype],
    d_inp2_ptr: MutKernelPtr[dtype],
    d_output_ptr: MutKernelPtr[dtype],
    residual_ptr: ImmutKernelPtr[dtype],
    gamma_ptr: ImmutKernelPtr[dtype],
    mean_ptr: ImmutKernelPtr[DType.float32],
    rstd_ptr: ImmutKernelPtr[DType.float32],
    d_gamma_ptr: MutKernelPtr[dtype],
    d_beta_ptr: MutKernelPtr[dtype],
    d_residual_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        layernorm_bwd[dtype, target](
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
        # GPU: fuse the broadcast accumulate into the d_input kernel itself
        # (layernorm_bwd_residual_gpu) instead of running layernorm_bwd's
        # plain d_input kernel followed by a separate broadcast sweep. This
        # is the ㉛-style fusion from the parity analysis: same dval
        # arithmetic, same RMW as the broadcast kernel's aligned=True path,
        # just done in the same pass so d_inp1/d_inp2 never round-trip
        # through the d_residual scratch plane.
        #
        # STORE_SCRATCH (stage 1 = True, stage 2 = False): stage 1 keeps
        # writing dval to d_residual_ptr (bit-identical to today, since the
        # value is never read back — the broadcast that used to read it is
        # now fused away). Stage 2 drops that store; see
        # _layernorm_bwd_residual_gpu's docstring for why that's safe.
        comptime STORE_SCRATCH = False
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        comptime width = 4
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_row_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        comptime dinput_kernel = layernorm_bwd_residual_gpu[
            dtype, BLOCK_SIZE, width, STORE_SCRATCH
        ]
        var dinput_compiled = device_ctx.compile_function[dinput_kernel]()
        device_ctx.enqueue_function(
            dinput_compiled,
            d_output_ptr,
            residual_ptr,
            gamma_ptr,
            mean_ptr,
            rstd_ptr,
            d_residual_ptr,
            d_inp1_ptr,
            d_inp2_ptr,
            batch_size,
            seq_len,
            channels,
            grid_dim=(num_row_blocks,),
            block_dim=(BLOCK_SIZE,),
        )

        _layernorm_dparam_gpu[dtype](
            d_output_ptr,
            residual_ptr,
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
        d_gamma: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        d_beta: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
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

        layernorm_bwd[dtype, target](
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
        d_gamma: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        d_beta: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
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

        layernorm_fused_residual_bwd[dtype, target](
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
            batch_size,
            seq_len,
            channels,
            ctx,
        )
