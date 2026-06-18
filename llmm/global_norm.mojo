import compiler
from std.sys import simd_width_of
from std.memory import alloc, UnsafePointer
from extensibility import InputTensor
from std.gpu.host import DeviceContext, DeviceAttribute
from std.math import sqrt, ceildiv, max, fma
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives import block

# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #


comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Helpers and Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
def _zero_gpu(
    idx: Int,
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
) -> None:
    (ptr + idx).store(0.0)


def zero_gpu(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    size: Int,
) -> None:
    # NOTE: This may be worth using in the future, also it could be an op in a mojo built-in library or even in tensor.
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        _zero_gpu(idx, ptr)


# =============================================================================
# Global Norm Squared - CPU
# =================================--------------------------------============


def global_norm_squared_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    data_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_params: Int,
) -> None:
    var num_workers = min(num_params, parallelism_level())
    var chunk = ceildiv(num_params, num_workers)
    var partials = alloc[Scalar[DType.float32]](num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, num_params - base)
        var worker_partial = partials + w
        # Zero the partial sum for the worker.
        worker_partial[0] = 0.0

        @always_inline
        def _simd[
            w_width: Int,
        ](local: Int) {data_ptr, base, worker_partial}:
            var val = (
                (data_ptr + base + local)
                .load[width=w_width]()
                .cast[DType.float32]()
            )
            worker_partial[0] += (val * val).reduce_add()

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_worker](num_workers)

    # Reduce the partial sums into the total sum at the end.
    var total_sum = Scalar[DType.float32](0.0)
    for w in range(num_workers):
        total_sum += partials[w]

    out_ptr[0] += total_sum
    partials.free()


# =============================================================================
# Global Norm Squared - GPU
# =============================================================================


@always_inline
def _global_norm_squared_for_range_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int,
](
    data_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    count: Int,
    tid: Int,
) -> Scalar[DType.float32]:
    var index = (block_idx.x * block_dim.x + thread_idx.x) * width
    var grid_width = block_dim.x * grid_dim.x * width

    var accumulator = SIMD[DType.float32, width](0.0)
    var accumulator_tail = Scalar[DType.float32](0.0)

    var idx = index
    while idx < count:
        if idx + width <= count:
            var val = (data_ptr + idx).load[width=width]().cast[DType.float32]()
            accumulator += val * val
        else:
            for j in range(idx, count):
                var val = data_ptr[j].cast[DType.float32]()
                accumulator_tail += val * val
        idx += grid_width

    var block_sum = block.sum[block_size=BLOCK_SIZE](
        accumulator.reduce_add() + accumulator_tail
    )
    return block_sum


def global_norm_squared_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    data_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    count: Int,
    stride: Int,
) -> None:
    var tid = Int(thread_idx.x)
    var block_sum = _global_norm_squared_for_range_gpu[
        dtype, BLOCK_SIZE, width
    ](data_ptr + Int(block_idx.y) * stride, count, tid)
    if tid == 0:
        var out_index = Int(block_idx.y * grid_dim.x + block_idx.x)
        out_ptr[out_index] += block_sum


def global_norm_aggregate_gpu[
    BLOCK_SIZE: Int,
](
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grid_size: Int,
) -> None:
    var tid = Int(thread_idx.x)
    var block_sum = Scalar[DType.float32](0.0)
    var idx = tid
    while idx < grid_size:
        block_sum += out_ptr[idx]
        idx += BLOCK_SIZE

    var total_sum = block.sum[block_size=BLOCK_SIZE](block_sum)
    if tid == 0:
        out_ptr[0] = total_sum


def global_norm_squared[
    dtype: DType,
    target: StaticString,
](
    output: MutableInputTensor[dtype=DType.float32, rank=1, static_spec=...],
    data: InputTensor[dtype=dtype, rank=2, static_spec=...],
    stride: Int64,
    num_slices: Int64,
    max_num_block_sums: Int64,
    reset: UInt32,
    ctx: DeviceContext,
) raises -> None:
    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        if reset != 0:
            output.unsafe_ptr()[0] = 0.0
        global_norm_squared_cpu[dtype, width](
            output.unsafe_ptr(), data.unsafe_ptr(), data.size()
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 512
        comptime RESIDENT_THREADS = 2048
        var device_ctx = ctx

        if reset != 0:
            var num_zero_threads = Int(max_num_block_sums)
            var num_zero_blocks = ceildiv(num_zero_threads, BLOCK_SIZE)
            comptime zero_kernel = zero_gpu
            var compiled_zero = device_ctx.compile_function[zero_kernel]()
            device_ctx.enqueue_function(
                compiled_zero,
                output.unsafe_ptr(),
                Int(max_num_block_sums),
                grid_dim=(num_zero_blocks,),
                block_dim=(BLOCK_SIZE,),
            )

        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var total_grid_size = (num_sm * RESIDENT_THREADS) // BLOCK_SIZE
        var grid_x = ceildiv(total_grid_size, Int(num_slices))
        var grid_y = Int(num_slices)
        var count = data.size() // Int(num_slices)

        comptime norm_kernel = global_norm_squared_gpu[dtype, BLOCK_SIZE, width]
        var compiled_norm = device_ctx.compile_function[norm_kernel]()
        device_ctx.enqueue_function(
            compiled_norm,
            output.unsafe_ptr(),
            data.unsafe_ptr(),
            count,
            Int(stride),
            grid_dim=(grid_x, grid_y),
            block_dim=(BLOCK_SIZE,),
        )

        var grid_size_actual = grid_x * grid_y
        comptime aggregate_kernel = global_norm_aggregate_gpu[BLOCK_SIZE]
        var compiled_aggregate = device_ctx.compile_function[aggregate_kernel]()
        device_ctx.enqueue_function(
            compiled_aggregate,
            output.unsafe_ptr(),
            grid_size_actual,
            grid_dim=(1,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("global_norm_squared")
struct GlobalNormSquared:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        data: InputTensor[dtype=dtype, rank=2, static_spec=...],
        stride: Int64,
        num_slices: Int64,
        max_num_block_sums: Int64,
        reset: UInt32,
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != 1:
            raise Error("output must have size 1")
        if data.size() != Int(num_slices) * Int(stride):
            raise Error("data must have size num_slices * stride")
        if max_num_block_sums < 0:
            raise Error("max_num_block_sums must be non-negative")
        if reset != 0 and reset != 1:
            raise Error("reset must be 0 or 1")

        global_norm_squared[dtype, target](
            output, data, stride, num_slices, max_num_block_sums, reset, ctx
        )
