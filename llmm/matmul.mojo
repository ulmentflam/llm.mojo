import compiler
from std.memory import alloc
from std.math import ceildiv
from layout import TileTensor
from linalg.matmul import matmul
from std.sys import simd_width_of
from std.gpu.primitives import block
from linalg.matmul.vendor import blas
from std.utils.index import IndexList
from extensibility import InputTensor
from linalg.transpose import transpose
from layout.tile_layout import row_major
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu.host import DeviceContext, DeviceAttribute
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from llmm.gelu import gelu, gelu_grad
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Matmul Forward
# ===----------------------------------------------------------------------=== #


def matmul_fwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool = False,
    has_bias: Bool = True,
](
    out_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    var rows = Int(batch_size * seq_len)  # M in matmul
    var in_channels = Int(channels)  # K in matmul
    var out_channels = Int(output_channels)  # N in matmul
    comptime elem_dtype = dtype

    var c = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=out_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var a = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )
    var b = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )

    @parameter
    @always_inline
    def epilogue_with_bias[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var offset = idx[0] * out_channels + idx[1]
        var v = (
            val.cast[DType.float32]()
            + (bias_ptr + idx[1]).load[width=width]().cast[DType.float32]()
        )
        comptime if use_gelu:
            (pre_gelu_ptr + offset).store(v.cast[elem_dtype]())
            (out_ptr + offset).store(
                gelu[DType.float32, width](v).cast[elem_dtype]()
            )
        else:
            (out_ptr + offset).store(v.cast[elem_dtype]())

    @parameter
    @always_inline
    def epilogue_no_bias[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var offset = idx[0] * out_channels + idx[1]
        var v = val.cast[DType.float32]()
        comptime if use_gelu:
            (pre_gelu_ptr + offset).store(v.cast[elem_dtype]())
            (out_ptr + offset).store(
                gelu[DType.float32, width](v).cast[elem_dtype]()
            )
        else:
            (out_ptr + offset).store(v.cast[elem_dtype]())

    comptime if has_bias:
        matmul[
            transpose_b=True,
            elementwise_lambda_fn=epilogue_with_bias,
            target=target,
        ](c, a, b, ctx=ctx)
    else:
        matmul[
            transpose_b=True,
            elementwise_lambda_fn=epilogue_no_bias,
            target=target,
        ](c, a, b, ctx=ctx)


@compiler.register("matmul_fwd")
struct MatmulFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_gelu: Bool = False,
        has_bias: Bool = True,
    ](
        output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        pre_gelu: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        weight: InputTensor[dtype=dtype, rank=2, static_spec=...],
        bias: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        output_channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != Int(batch_size * seq_len * output_channels):
            raise Error(
                "output must have the same size as batch_size * seq_len *"
                " output_channels"
            )
        # The use_gelu=False instantiation contains no stores to pre_gelu
        # (comptime-dead code), so a dummy buffer of any size is sound there.
        comptime if use_gelu:
            if pre_gelu.size() != Int(batch_size * seq_len * output_channels):
                raise Error(
                    "pre_gelu must have the same size as batch_size * seq_len"
                    " * output_channels"
                )
        if x.size() != Int(batch_size * seq_len * channels):
            raise Error(
                "input must have the same size as batch_size * seq_len *"
                " channels"
            )
        if weight.size() != Int(output_channels * channels):
            raise Error(
                "weight must have the same size as output_channels * channels"
            )
        comptime if has_bias:
            if bias.size() != Int(output_channels):
                raise Error("bias must have the same size as output_channels")

        matmul_fwd[
            dtype,
            target,
            use_gelu=use_gelu,
            has_bias=has_bias,
        ](
            output.unsafe_ptr(),
            pre_gelu.unsafe_ptr(),
            x.unsafe_ptr(),
            weight.unsafe_ptr(),
            bias.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            output_channels,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Matmul Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _matmul_bias_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    accumulate: Bool,
    width: Int,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
    num_tiles: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
) -> None:
    for tile in range(block_row, num_tiles, stride):
        var base = tile * width
        if base + width <= out_channels:
            var accumulator = SIMD[DType.float32, width](0.0)
            for r in range(tid, rows, BLOCK_SIZE):
                accumulator += (
                    (d_output_ptr + r * out_channels + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
            var tile_sum = block.sum[block_size=BLOCK_SIZE](accumulator)
            if tid == 0:
                comptime if accumulate:
                    var previous = (
                        (d_bias_ptr + base)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    (d_bias_ptr + base).store(
                        (previous + tile_sum).cast[dtype]()
                    )
                else:
                    (d_bias_ptr + base).store(tile_sum.cast[dtype]())
        else:
            # Ragged edge of the last tile, scalar steps. Loop bounds depend
            # only on `base`, so every thread reaches block.sum uniformly.
            for c in range(base, out_channels):
                var accumulator = Scalar[DType.float32](0.0)
                for r in range(tid, rows, BLOCK_SIZE):
                    accumulator += d_output_ptr[r * out_channels + c].cast[
                        DType.float32
                    ]()
                var col_sum = block.sum[block_size=BLOCK_SIZE](accumulator)
                if tid == 0:
                    comptime if accumulate:
                        var previous = d_bias_ptr[c].cast[DType.float32]()
                        d_bias_ptr[c] = (previous + col_sum).cast[dtype]()
                    else:
                        d_bias_ptr[c] = col_sum.cast[dtype]()


def matmul_bias_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    accumulate: Bool,
    width: Int = 4,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    output_channels: Int64,
) -> None:
    var rows = Int(batch_size * seq_len)
    var out_channels = Int(output_channels)
    var num_tiles = ceildiv(out_channels, width)
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x)
    var block_row = Int(block_idx.x)

    _matmul_bias_bwd_gpu[dtype, BLOCK_SIZE, accumulate, width](
        d_bias_ptr,
        d_output_ptr,
        rows,
        out_channels,
        num_tiles,
        tid,
        stride,
        block_row,
    )


@always_inline
def matmul_bias_bwd_cpu[
    dtype: DType,
    width: Int,
    accumulate: Bool,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,  # (B * T) or (batch_size * seq_len)
    out_channels: Int,  # OC
) -> None:
    var num_workers = min(out_channels, parallelism_level())
    var cols_per_worker = ceildiv(out_channels, num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * cols_per_worker
        var count = min(cols_per_worker, out_channels - base)

        @always_inline
        def _simd[
            w: Int,
        ](local: Int) {d_bias_ptr, d_output_ptr, base, rows, out_channels,}:
            var idx = base + local
            var accumulator = SIMD[DType.float32, w](0.0)
            for r in range(rows):
                var offset = r * out_channels + idx
                accumulator += (
                    (d_output_ptr + offset)
                    .load[width=w]()
                    .cast[DType.float32]()
                )
            comptime if accumulate:
                var previous = (
                    (d_bias_ptr + idx).load[width=w]().cast[DType.float32]()
                )
                (d_bias_ptr + idx).store((previous + accumulator).cast[dtype]())
            else:
                (d_bias_ptr + idx).store(accumulator.cast[dtype]())

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_worker](num_workers)


def matmul_bias_bwd[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[DType.float32]()
        matmul_bias_bwd_cpu[dtype, simd_width, accumulate](
            d_bias_ptr,
            d_output_ptr,
            Int(batch_size * seq_len),
            Int(output_channels),
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        comptime width = 4
        var device_ctx = ctx
        var num_tiles = ceildiv(Int(output_channels), width)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_tiles, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = matmul_bias_bwd_gpu[
            dtype, BLOCK_SIZE, accumulate, width
        ]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            d_bias_ptr,
            d_output_ptr,
            batch_size,
            seq_len,
            output_channels,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


def matmul_d_input_bwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool,
](
    d_input_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    """Computes d_input = d_output @ weight. Overwrites d_input: llm.c overwrites
    activation grads; only weight/bias grads accumulate across micro-steps."""
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)
    comptime elem_dtype = dtype

    var c_d_input = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=d_input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )
    var a_d_output = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=d_output_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var b_weight = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )

    comptime if use_gelu:

        @parameter
        @always_inline
        def d_input_epilogue[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
            var offset = idx[0] * in_channels + idx[1]
            var v = val.cast[DType.float32]()
            var pre_gelu = (
                (pre_gelu_ptr + offset)
                .load[width=width]()
                .cast[DType.float32]()
            )
            v *= gelu_grad[DType.float32, width](pre_gelu)
            (d_input_ptr + offset).store(v.cast[elem_dtype]())

        matmul[
            transpose_b=False,
            elementwise_lambda_fn=d_input_epilogue,
            target=target,
        ](c_d_input, a_d_output, b_weight, ctx=ctx)
    else:
        matmul[transpose_b=False, target=target](
            c_d_input, a_d_output, b_weight, ctx=ctx
        )


@always_inline
def _add_into[
    dtype: DType,
    width: Int,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: MutKernelPtr[dtype],
    total: Int,
) -> None:
    """dst += src elementwise, f32 math, one rounding at the store."""
    var num_workers = min(total, parallelism_level())
    var chunk = ceildiv(total, num_workers)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, total - base)

        @always_inline
        def _simd[
            w_: Int,
        ](local: Int) {dst_ptr, src_ptr, base}:
            var idx = base + local
            var a = (dst_ptr + idx).load[width=w_]().cast[DType.float32]()
            var b = (src_ptr + idx).load[width=w_]().cast[DType.float32]()
            (dst_ptr + idx).store((a + b).cast[dtype]())

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_worker](num_workers)


def matmul_d_weight_bwd[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_weight_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    scratch_ptr: MutKernelPtr[dtype],
    batch_size: Int64,  # B
    seq_len: Int64,  # T
    channels: Int64,  # C
    output_channels: Int64,  # OC
    ctx: DeviceContext,
) raises -> None:
    """Computes d_weight = d_output^T @ input. linalg.matmul rejects transpose_a (#6626), so
    GPU goes through the vendor BLAS (transposed A is native and beta folds
    in accumulation) and CPU materializes d_output^T once into scratch
    (O(rows * OC) traffic next to the GEMM's O(2 * rows * OC * C) flops)."""
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    var c_d_weight = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=d_weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )
    var a_d_output = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=d_output_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var b_input = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )

    comptime if is_gpu[target]():
        # Backend resolves per vendor.
        blas.matmul(
            ctx,
            c_d_weight,
            a_d_output,
            b_input,
            c_row_major=True,
            transpose_a=True,
            beta=Float32(1.0) if accumulate else Float32(0.0),
        )
    else:
        var scratch_t = TileTensor(
            Span[Scalar[dtype], MutAnyOrigin](
                ptr=scratch_ptr, length=out_channels * rows
            ),
            row_major(out_channels, rows),
        )
        var perms = alloc[Scalar[DType.int]](2)
        perms[0] = 1
        perms[1] = 0
        transpose(scratch_t, a_d_output, perms)
        perms.free()

        comptime if accumulate:
            # NOT an epilogue += : on the f32 Apple Accelerate path the
            # elementwise lambda runs as a sweep AFTER cblas has already
            # overwritten C, so "load previous" there reads the fresh GEMM
            # result and doubles it. Materialize, then add.
            var temp = alloc[Scalar[dtype]](out_channels * in_channels)
            var c_temp = TileTensor(
                Span[Scalar[dtype], MutAnyOrigin](
                    ptr=temp.as_unsafe_any_origin(),
                    length=out_channels * in_channels,
                ),
                row_major(out_channels, in_channels),
            )
            matmul[transpose_b=False, target=target](
                c_temp, scratch_t, b_input, ctx=ctx
            )
            comptime simd_width = simd_width_of[DType.float32]()
            _add_into[dtype, simd_width](
                d_weight_ptr,
                rebind[MutKernelPtr[dtype]](temp.as_unsafe_any_origin()),
                out_channels * in_channels,
            )
            temp.free()
        else:
            matmul[transpose_b=False, target=target](
                c_d_weight, scratch_t, b_input, ctx=ctx
            )


def matmul_bwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool = False,
    accumulate: Bool = True,
    has_bias: Bool = True,
](
    d_input_ptr: MutKernelPtr[dtype],
    d_weight_ptr: MutKernelPtr[dtype],
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    scratch_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    comptime if has_bias:
        matmul_bias_bwd[dtype, target, accumulate](
            d_bias_ptr,
            d_output_ptr,
            batch_size,
            seq_len,
            output_channels,
            ctx,
        )
    matmul_d_input_bwd[dtype, target, use_gelu](
        d_input_ptr,
        d_output_ptr,
        weight_ptr,
        pre_gelu_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
    )
    matmul_d_weight_bwd[dtype, target, accumulate](
        d_weight_ptr,
        d_output_ptr,
        input_ptr,
        scratch_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
    )


# ===----------------------------------------------------------------------=== #
# Matmul Backward Compiler Registration
# ===----------------------------------------------------------------------=== #


@always_inline
def _check_bwd_sizes[
    has_bias: Bool = True,
](
    d_input_size: Int,
    d_weight_size: Int,
    d_bias_size: Int,
    d_output_size: Int,
    x_size: Int,
    weight_size: Int,
    scratch_size: Int,
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
) raises -> None:
    var rows_x_channels = Int(batch_size * seq_len * channels)
    var rows_x_out = Int(batch_size * seq_len * output_channels)
    var weight_elems = Int(output_channels * channels)
    if d_input_size != rows_x_channels:
        raise Error(
            "d_input must have the same size as batch_size * seq_len * channels"
        )
    if d_weight_size != weight_elems:
        raise Error(
            "d_weight must have the same size as output_channels * channels"
        )
    comptime if has_bias:
        if d_bias_size != Int(output_channels):
            raise Error("d_bias must have the same size as output_channels")
    if d_output_size != rows_x_out:
        raise Error(
            "d_output must have the same size as batch_size * seq_len *"
            " output_channels"
        )
    if x_size != rows_x_channels:
        raise Error(
            "input must have the same size as batch_size * seq_len * channels"
        )
    if weight_size != weight_elems:
        raise Error(
            "weight must have the same size as output_channels * channels"
        )
    if scratch_size != rows_x_out:
        raise Error(
            "scratch must have the same size as batch_size * seq_len *"
            " output_channels"
        )


@compiler.register("matmul_bwd")
struct MatmulBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_gelu: Bool = False,
        accumulate: Bool = True,
        has_bias: Bool = True,
    ](
        d_input: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_weight: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_bias: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        scratch: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_output: InputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        weight: InputTensor[dtype=dtype, rank=2, static_spec=...],
        pre_gelu: InputTensor[dtype=dtype, rank=2, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        output_channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        _check_bwd_sizes[has_bias=has_bias](
            d_input.size(),
            d_weight.size(),
            d_bias.size(),
            d_output.size(),
            x.size(),
            weight.size(),
            scratch.size(),
            batch_size,
            seq_len,
            channels,
            output_channels,
        )
        # The use_gelu=False instantiation contains no loads from pre_gelu
        # (comptime-dead code), so a dummy buffer of any size is sound there.
        # pre_gelu here is the pre-activation of this matmul's INPUT (llm.c
        # composition: d_input = (d_output @ W) * gelu'(pre_gelu)), so it has
        # d_input's shape, not the forward's (B*T, OC).
        comptime if use_gelu:
            if pre_gelu.size() != Int(batch_size * seq_len * channels):
                raise Error(
                    "pre_gelu must have the same size as batch_size * seq_len"
                    " * channels"
                )
        matmul_bwd[
            dtype,
            target,
            use_gelu=use_gelu,
            accumulate=accumulate,
            has_bias=has_bias,
        ](
            d_input.unsafe_ptr(),
            d_weight.unsafe_ptr(),
            d_bias.unsafe_ptr(),
            d_output.unsafe_ptr(),
            x.unsafe_ptr(),
            weight.unsafe_ptr(),
            pre_gelu.unsafe_ptr(),
            scratch.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            output_channels,
            ctx,
        )
