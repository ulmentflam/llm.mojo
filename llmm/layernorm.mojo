import compiler
from tensor import InputTensor
from std.sys import simd_width_of
from std.gpu.primitives import block
from std.gpu.host import DeviceAttribute
from std.memory import alloc, UnsafePointer
from std.gpu.host.info import is_cpu, is_gpu
from std.math import fma, sqrt, ceildiv, rsqrt
from std.algorithm import vectorize, sync_parallelize
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.runtime.asyncrt import DeviceContextPtr, parallelism_level

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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    epsilon: Scalar[DType.float32],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    epsilon: Scalar[DType.float32],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
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

    sync_parallelize[_worker](num_workers)


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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    epsilon: Scalar[DType.float32],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    epsilon: Scalar[DType.float32],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    epsilon: Scalar[DType.float32],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContextPtr,
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
        var device_ctx = ctx.get_device_context()
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = layernorm_fwd_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[
            func=gpu_kernel, signature_func=gpu_kernel
        ]()
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
        ctx: DeviceContextPtr,
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
# LayerNorm Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _layernorm_bwd_cpu[
    dtype: DType,
    width: Int,
](
    idx: Int,
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dgamma_partial_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dbeta_partial_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_gamma_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    d_beta_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
) -> None:
    var total = Int(batch_size * seq_len)
    var c = Int(channels)
    var num_workers = min(total, parallelism_level())
    var rows_per_worker = ceildiv(total, num_workers)

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
                dgamma_row,
                dbeta_row,
                c,
            )

    sync_parallelize[_worker](num_workers)

    # Reduce the per-worker partials into the parameter gradients. We add into
    # whatever is already there (the atomic path accumulated the same way), so
    # callers that accumulate across micro-steps keep that behavior.
    for j in range(c):
        var acc_dgamma = Scalar[DType.float32](0.0)
        var acc_dbeta = Scalar[DType.float32](0.0)
        for w in range(num_workers):
            acc_dgamma += dgamma_partial[w * c + j]
            acc_dbeta += dbeta_partial[w * c + j]
        d_gamma_ptr[j] += acc_dgamma
        d_beta_ptr[j] += acc_dbeta

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
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    channels: Int,
) -> None:
    # d_input only: one block reduces a row's statistics with block.sum and
    # writes that row's (disjoint) d_input slice, so there is nothing to race.
    # The parameter gradients reduce over the opposite axis (all rows) and are
    # handled by _layernorm_dgamma_dbeta_gpu, which needs no atomics either.
    comptime BLOCK_SPAN = BLOCK_SIZE * width

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
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var x = (
                    (input_ptr + idx).load[width=width]().cast[DType.float32]()
                )
                var g = (
                    (gamma_ptr + i).load[width=width]().cast[DType.float32]()
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
                    .load[width=width]()
                    .cast[DType.float32]()
                )
                var x = (
                    (input_ptr + idx).load[width=width]().cast[DType.float32]()
                )
                var g = (
                    (gamma_ptr + i).load[width=width]().cast[DType.float32]()
                )
                var x_hat = (x - mean_vec) * rstd_vec
                var d_input = rstd_vec * (
                    g * dy - (sum_gdy * inv_c) - (x_hat * sum_gdy_xhat * inv_c)
                )
                (d_input_ptr + idx).store(d_input.cast[dtype]())
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
    d_output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_gamma_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    d_beta_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
            var sum_dgamma = block.sum[block_size=BLOCK_SIZE](acc_dgamma)
            var sum_dbeta = block.sum[block_size=BLOCK_SIZE](acc_dbeta)
            if tid == 0:
                # Accumulate into existing grads (matches the CPU path and the
                # old atomic accumulate-into-buffer behavior).
                var prev_dgamma = (d_gamma_ptr + base).load[width=width]()
                var prev_dbeta = (d_beta_ptr + base).load[width=width]()
                (d_gamma_ptr + base).store(prev_dgamma + sum_dgamma)
                (d_beta_ptr + base).store(prev_dbeta + sum_dbeta)
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
                    d_gamma_ptr[c] += sum_dgamma
                    d_beta_ptr[c] += sum_dbeta


def layernorm_dgamma_dbeta_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
](
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_gamma_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    d_beta_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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


def layernorm_bwd[
    dtype: DType,
    target: StaticString,
](
    d_output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    d_input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_gamma_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    d_beta_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    ctx: DeviceContextPtr,
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
        var device_ctx = ctx.get_device_context()
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        # Kernel 1: d_input, one block per row tile (reduces over channels).
        var num_row_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        comptime dinput_kernel = layernorm_bwd_gpu[dtype, BLOCK_SIZE, width]
        var dinput_compiled = device_ctx.compile_function[
            func=dinput_kernel, signature_func=dinput_kernel
        ]()
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

        # Kernel 2: d_gamma/d_beta, one block per channel tile (reduces over
        # rows). Separate launch so each gradient runs on its natural axis.
        var num_chan_tiles = ceildiv(Int(channels), width)
        var num_chan_blocks = max(
            min(num_chan_tiles, SM_OVERPROVISION * num_sm), 1
        )
        comptime dparam_kernel = layernorm_dgamma_dbeta_gpu[
            dtype, BLOCK_SIZE, width
        ]
        var dparam_compiled = device_ctx.compile_function[
            func=dparam_kernel, signature_func=dparam_kernel
        ]()
        device_ctx.enqueue_function(
            dparam_compiled,
            d_output_ptr,
            input_ptr,
            mean_ptr,
            rstd_ptr,
            d_gamma_ptr,
            d_beta_ptr,
            batch_size,
            seq_len,
            channels,
            grid_dim=(num_chan_blocks,),
            block_dim=(BLOCK_SIZE,),
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
        d_gamma: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_beta: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        channels: Int64,  # Our C
        ctx: DeviceContextPtr,
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
