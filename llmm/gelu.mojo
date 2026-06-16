import compiler
from std.sys import simd_width_of
from std.memory import UnsafePointer
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
from std.math import sqrt, ceildiv, tanh, pi
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import block_dim, block_idx, grid_dim, thread_idx


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4
comptime GELU_CONSTANT = 0.044715
comptime SQRT_TWO_OVER_PI = sqrt(2.0 / pi)

# ===----------------------------------------------------------------------=== #
# GELU Forward
# ===----------------------------------------------------------------------=== #


@always_inline
def gelu[
    dtype: DType,
    width: Int,
](x: SIMD[dtype, width],) -> SIMD[dtype, width]:
    comptime assert (
        dtype.is_floating_point()
    ), "gelu requires a floating point dtype"
    var u = SIMD[dtype, width](SQRT_TWO_OVER_PI) * (
        x + GELU_CONSTANT * x * x * x
    )
    var tanh_u = tanh(u)
    return 0.5 * x * (1.0 + tanh_u)


@always_inline
def _gelu_fwd[
    dtype: DType,
    width: Int,
](
    idx: Int,
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
) -> None:
    var x = (x_ptr + idx).load[width=width]().cast[DType.float32]()
    (out_ptr + idx).store(gelu(x).cast[dtype]())


def gelu_fwd_gpu[
    dtype: DType,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_params: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _gelu_fwd[dtype, width](idx, out_ptr, x_ptr)
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _gelu_fwd[dtype, 1](i, out_ptr, x_ptr)


def gelu_fwd_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_params: Int,
) -> None:
    var num_workers = min(num_params, parallelism_level())
    var chunk = ceildiv(num_params, num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, num_params - base)

        @always_inline
        def _simd[
            w: Int,
        ](local: Int) {out_ptr, x_ptr, base}:
            _gelu_fwd[dtype, w](base + local, out_ptr, x_ptr)

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_worker](num_workers)


def gelu_fwd[
    dtype: DType,
    target: StaticString,
](
    output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
    x: InputTensor[dtype=dtype, rank=2, static_spec=...],
    ctx: DeviceContext,
) raises -> None:
    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        gelu_fwd_cpu[dtype, width](
            output.unsafe_ptr(), x.unsafe_ptr(), x.size()
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        var device_ctx = ctx
        # Each thread handles `width` elements with no grid-stride loop, so
        # the grid must cover every width-wide tile.
        var num_threads = ceildiv(x.size(), width)
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)

        comptime gpu_kernel = gelu_fwd_gpu[dtype, width]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            output.unsafe_ptr(),
            x.unsafe_ptr(),
            x.size(),
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("gelu_fwd")
struct GeluFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != x.size():
            raise Error("output and x must have the same size")
        gelu_fwd[dtype, target](output, x, ctx)


# ===----------------------------------------------------------------------=== #
# GELU Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def gelu_grad[
    dtype: DType,
    width: Int,
](x: SIMD[dtype, width],) -> SIMD[dtype, width]:
    comptime assert (
        dtype.is_floating_point()
    ), "gelu_grad requires a floating point dtype"
    var u = SIMD[dtype, width](SQRT_TWO_OVER_PI) * (
        x + GELU_CONSTANT * x * x * x
    )
    var tanh_u = tanh(u)
    var du_dx = SIMD[dtype, width](SQRT_TWO_OVER_PI) * (
        1.0 + 3.0 * GELU_CONSTANT * x * x
    )
    return 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u * tanh_u) * du_dx


@always_inline
def _gelu_bwd[
    dtype: DType,
    width: Int,
](
    idx: Int,
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
) -> None:
    var x = (x_ptr + idx).load[width=width]().cast[DType.float32]()
    (out_ptr + idx).store(gelu_grad(x).cast[dtype]())


def gelu_bwd_gpu[
    dtype: DType,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_params: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _gelu_bwd[dtype, width](idx, out_ptr, x_ptr)
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _gelu_bwd[dtype, 1](i, out_ptr, x_ptr)


def gelu_bwd_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_params: Int,
) -> None:
    var num_workers = min(num_params, parallelism_level())
    var chunk = ceildiv(num_params, num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, num_params - base)

        # NOTE: I've written this block enough times now thia should be it's own helper function.
        @always_inline
        def _simd[
            w: Int,
        ](local: Int) {out_ptr, x_ptr, base}:
            _gelu_bwd[dtype, w](base + local, out_ptr, x_ptr)

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_worker](num_workers)


def gelu_bwd[
    dtype: DType,
    target: StaticString,
](
    output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
    x: InputTensor[dtype=dtype, rank=2, static_spec=...],
    ctx: DeviceContext,
) raises -> None:
    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        gelu_bwd_cpu[dtype, width](
            output.unsafe_ptr(), x.unsafe_ptr(), x.size()
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        var device_ctx = ctx
        # Each thread handles `width` elements with no grid-stride loop, so
        # the grid must cover every width-wide tile.
        var num_threads = ceildiv(x.size(), width)
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)

        comptime gpu_kernel = gelu_bwd_gpu[dtype, width]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            output.unsafe_ptr(),
            x.unsafe_ptr(),
            x.size(),
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("gelu_bwd")
struct GeluBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != x.size():
            raise Error("output and x must have the same size")
        gelu_bwd[dtype, target](output, x, ctx)
