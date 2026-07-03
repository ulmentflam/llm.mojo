import compiler
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
from std.sys import simd_width_of, align_of
from std.math import sqrt, ceildiv, tanh, pi
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
comptime GELU_CONSTANT = 0.044715
comptime SQRT_TWO_OVER_PI = 0.7978845608028654
# NOTE: Apple Silicon has issues with comptime computing sqrt(2.0 / pi) so we hardcode the value.
# comptime SQRT_TWO_OVER_PI = sqrt(2.0 / pi)


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
    aligned: Bool = False,
](
    idx: Int,
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
) -> None:
    # `aligned` is opt-in: the CPU caller's worker-chunk offset isn't
    # provably width-aligned (chunk = ceildiv(num_params, max_workers) is a
    # runtime value, not a multiple of width in general), but the GPU
    # caller's idx = global_tid*width IS (same proof as adamw's idx).
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        var x = (
            (x_ptr + idx)
            .load[width=width, alignment=align]()
            .cast[DType.float32]()
        )
        (out_ptr + idx).store[width=width, alignment=align](
            gelu(x).cast[dtype]()
        )
    else:
        var x = (x_ptr + idx).load[width=width]().cast[DType.float32]()
        (out_ptr + idx).store(gelu(x).cast[dtype]())


def gelu_fwd_gpu[
    dtype: DType,
    width: Int,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _gelu_fwd[dtype, width, aligned=True](idx, out_ptr, x_ptr)
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _gelu_fwd[dtype, 1](i, out_ptr, x_ptr)


def gelu_fwd_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
) raises -> None:
    var max_workers = parallelism_level()
    var chunk = ceildiv(num_params, max_workers)
    var num_workers = ceildiv(num_params, chunk)

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

    traced_parallelize["gelu_fwd", _worker](num_workers)


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


# ===----------------------------------------------------------------------=== #
# Fused bias (+ optional GELU) epilogue — Metal route-around for matmul_fwd.
#
# GOTCHA (Metal): linalg.matmul's fused `elementwise_lambda_fn` epilogue
# produces WRONG results on Apple GPU (probe_matmul_bias.mojo: has_bias=False
# passes with max_abs=0, has_bias=True fails with max_abs≈74; see also
# docs/ai/ai_assisted_optimizations_and_benchmarks.md). Root cause is a bug in
# how the Metal backend lowers the elementwise lambda into the GEMM kernel.
#
# On NVIDIA, matmul_fwd uses the cuBLASLt BIAS (and optionally GELU_AUX_BIAS)
# epilogue directly inside the GEMM, so this function is not called from the
# NVIDIA path when USE_GELU_FUSION=True (see matmul.mojo). On Metal, matmul_fwd
# runs a plain GEMM (no epilogue) and then calls bias_gelu_fwd in a separate
# pass to apply the bias and GELU, matching the CPU epilogue exactly.
# ===----------------------------------------------------------------------=== #


@always_inline
def _bias_gelu_kernel_vec[
    dtype: DType,
    has_bias: Bool,
    use_gelu: Bool,
    vec: Int,
](
    out_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: MutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
) -> None:
    # 2D grid following the encoder_fwd_gpu_kernel convention:
    #   grid_dim = (rows, num_col_tiles),  block_dim = (BLOCK_SIZE,)
    #   block_idx.x = row,  block_idx.y = column-tile
    # col_base is the first column this thread owns; the bias index equals
    # col_base with NO i % out_channels modulo — the row coordinate lives in
    # block_idx.x, not encoded in a flat linear index.
    var row = Int(block_idx.x)
    var col_base = (
        Int(block_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    ) * vec
    if row >= rows or col_base >= out_channels:
        return
    var i_base = row * out_channels + col_base
    if col_base + vec <= out_channels:
        # Full vector: all vec elements are within the row.
        var v = (out_ptr + i_base).load[width=vec]().cast[DType.float32]()
        comptime if has_bias:
            v = (
                v
                + (bias_ptr + col_base).load[width=vec]().cast[DType.float32]()
            )
        comptime if use_gelu:
            (pre_gelu_ptr + i_base).store(v.cast[dtype]())
            (out_ptr + i_base).store(gelu[DType.float32, vec](v).cast[dtype]())
        else:
            (out_ptr + i_base).store(v.cast[dtype]())
    else:
        # Tail: fewer than vec columns remain before the row boundary.
        # Guarded generically; never reached when out_channels % vec == 0
        # (all real OC values — 768, 2304, 3072 — are divisible by 8).
        for k in range(out_channels - col_base):
            var v = (out_ptr + i_base + k)[].cast[DType.float32]()
            comptime if has_bias:
                v = v + (bias_ptr + col_base + k)[].cast[DType.float32]()
            comptime if use_gelu:
                (pre_gelu_ptr + i_base + k)[] = v.cast[dtype]()
                (out_ptr + i_base + k)[] = gelu[DType.float32, 1](v).cast[
                    dtype
                ]()
            else:
                (out_ptr + i_base + k)[] = v.cast[dtype]()


def bias_gelu_fwd[
    dtype: DType,
    target: StaticString,
    has_bias: Bool = True,
    use_gelu: Bool = False,
](
    out_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: MutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
    ctx: DeviceContext,
) raises -> None:
    var n = rows * out_channels
    comptime if is_cpu[target]():
        for i in range(n):
            var v = out_ptr[i].cast[DType.float32]()
            comptime if has_bias:
                v = v + bias_ptr[i % out_channels].cast[DType.float32]()
            comptime if use_gelu:
                pre_gelu_ptr[i] = v.cast[dtype]()
                out_ptr[i] = gelu[DType.float32, 1](v).cast[dtype]()
            else:
                out_ptr[i] = v.cast[dtype]()
    elif is_gpu[target]():
        # Vectorized 2D-grid dispatch: rows × column-tiles, each thread
        # handles `VEC` contiguous columns.  Eliminates the per-element
        # `i % out_channels` modulo (row index lives in block_idx.x) and
        # issues vec-wide SIMD loads/stores instead of scalar ones.
        comptime BLOCK_SIZE = 256
        comptime VEC = simd_width_of[dtype]()
        var device_ctx = ctx
        var num_col_tiles = ceildiv(out_channels, BLOCK_SIZE * VEC)
        comptime gpu_kernel = _bias_gelu_kernel_vec[
            dtype, has_bias, use_gelu, VEC
        ]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            out_ptr,
            pre_gelu_ptr,
            bias_ptr,
            rows,
            out_channels,
            grid_dim=(rows, num_col_tiles),
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
    aligned: Bool = False,
](
    idx: Int,
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
) -> None:
    # `aligned` is opt-in — see _gelu_fwd for the CPU-vs-GPU offset proof.
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        var x = (
            (x_ptr + idx)
            .load[width=width, alignment=align]()
            .cast[DType.float32]()
        )
        (out_ptr + idx).store[width=width, alignment=align](
            gelu_grad(x).cast[dtype]()
        )
    else:
        var x = (x_ptr + idx).load[width=width]().cast[DType.float32]()
        (out_ptr + idx).store(gelu_grad(x).cast[dtype]())


def gelu_bwd_gpu[
    dtype: DType,
    width: Int,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _gelu_bwd[dtype, width, aligned=True](idx, out_ptr, x_ptr)
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _gelu_bwd[dtype, 1](i, out_ptr, x_ptr)


def gelu_bwd_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
) raises -> None:
    var max_workers = parallelism_level()
    var chunk = ceildiv(num_params, max_workers)
    var num_workers = ceildiv(num_params, chunk)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, num_params - base)

        @always_inline
        def _simd[
            w: Int,
        ](local: Int) {out_ptr, x_ptr, base}:
            _gelu_bwd[dtype, w](base + local, out_ptr, x_ptr)

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    traced_parallelize["gelu_bwd", _worker](num_workers)


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
