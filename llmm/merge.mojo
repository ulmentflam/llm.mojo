import compiler
from std.sys import simd_width_of
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.algorithm import sync_parallelize
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from llmm.profiler import traced_parallelize

from llmm.head_layout import (
    head_layout_flat_at_head_dim_zero,
    layout_copy,
    token_layout_plane0_flat_at_head_dim_zero,
    token_layout_plane0_flat_from_head_layout_flat,
    vectorize_layout_copy,
)
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime MERGE_GPU_WIDTH = 1
comptime MERGE_GPU_STREAMING = False


# ===----------------------------------------------------------------------=== #
# Merge helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _merge_cpu_tile[
    dtype: DType,
    width: Int,
    backward: Bool,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    bh: Int,
    t: Int,
) -> None:
    var channels = num_heads * head_dim
    var b = bh // num_heads
    var h = bh % num_heads
    var head_layout_src_flat_at_head_dim_zero = (
        head_layout_flat_at_head_dim_zero(b, h, t, seq_len, num_heads, head_dim)
    )
    var token_layout_dst_flat_at_head_dim_zero = (
        token_layout_plane0_flat_at_head_dim_zero(
            b, t, h, seq_len, 1, channels, head_dim
        )
    )
    vectorize_layout_copy[dtype, width, backward](
        dst_ptr,
        src_ptr,
        token_layout_dst_flat_at_head_dim_zero,
        head_layout_src_flat_at_head_dim_zero,
        head_dim,
    )


@always_inline
def _merge_cpu[
    dtype: DType,
    width: Int,
    backward: Bool,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) raises -> None:
    @parameter
    def _worker(bh: Int):
        for t in range(seq_len):
            _merge_cpu_tile[dtype, width, backward](
                dst_ptr,
                src_ptr,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                bh,
                t,
            )

    traced_parallelize["merge", _worker](batch_size * num_heads)


# ===----------------------------------------------------------------------=== #
# Merge Forward
# ===----------------------------------------------------------------------=== #


def merge_fwd_cpu[
    dtype: DType,
    width: Int,
](
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) raises -> None:
    _merge_cpu[dtype, width, backward=False](
        dst_ptr, src_ptr, batch_size, seq_len, num_heads, head_dim
    )


def merge_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    backward: Bool,
    width: Int = 1,
    streaming: Bool = False,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    # width=1/streaming=False: one element per thread (already coalesced, max
    # parallelism — vectorizing to width=8 was measured slower; see
    # split.mojo::split_gpu). width>1 threads each own a `width`-wide
    # head_dim-contiguous chunk instead (item 4 experiment) — safe because
    # head_dim is always a multiple of the widths swept here (GPT-2 head_dim
    # is 64) so a chunk never straddles the head_dim boundary where the
    # head-layout -> token-layout mapping stops being +1-contiguous.
    var chunk_index = Int(block_idx.x * block_dim.x + thread_idx.x)
    var head_layout_flat_index = chunk_index * width
    var num_elements = Int(batch_size * num_heads * seq_len * head_dim)
    if head_layout_flat_index < num_elements:
        var token_layout_dst_flat_index = (
            token_layout_plane0_flat_from_head_layout_flat(
                head_layout_flat_index,
                Int(head_dim),
                Int(seq_len),
                Int(num_heads),
                1,
                Int(channels),
            )
        )
        layout_copy[dtype, width, backward, streaming](
            dst_ptr,
            src_ptr,
            token_layout_dst_flat_index,
            head_layout_flat_index,
        )


def merge_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    merge_gpu[
        dtype,
        BLOCK_SIZE,
        backward=False,
        width=MERGE_GPU_WIDTH,
        streaming=MERGE_GPU_STREAMING,
    ](
        dst_ptr,
        src_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        channels,
    )


def merge_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    d_dst_ptr: ImmutKernelPtr[dtype],
    d_src_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    merge_gpu[
        dtype,
        BLOCK_SIZE,
        backward=True,
        width=MERGE_GPU_WIDTH,
        streaming=MERGE_GPU_STREAMING,
    ](
        d_src_ptr,
        d_dst_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        channels,
    )


def merge_fwd[
    dtype: DType,
    target: StaticString,
](
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises:
    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        merge_fwd_cpu[dtype, width](
            src_ptr,
            dst_ptr,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        )
    elif is_gpu[target]():
        var channels = num_heads * head_dim
        var num_elements = batch_size * num_heads * seq_len * head_dim
        comptime BLOCK_SIZE = 256
        var num_chunks = (num_elements + MERGE_GPU_WIDTH - 1) // MERGE_GPU_WIDTH
        var num_blocks = (num_chunks + BLOCK_SIZE - 1) // BLOCK_SIZE
        comptime gpu_kernel = merge_fwd_gpu[dtype, BLOCK_SIZE]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            src_ptr,
            dst_ptr,
            Int64(batch_size),
            Int64(seq_len),
            Int64(num_heads),
            Int64(head_dim),
            Int64(channels),
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("merge_fwd")
struct MergeFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        dst: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        src: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        num_heads: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        merge_fwd[dtype, target](
            src.unsafe_ptr(),
            dst.unsafe_ptr(),
            Int(batch_size),
            Int(seq_len),
            Int(num_heads),
            Int(head_dim),
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Merge Backward
# ===----------------------------------------------------------------------=== #


def merge_bwd_cpu[
    dtype: DType,
    width: Int,
](
    d_dst_ptr: ImmutKernelPtr[dtype],
    d_src_ptr: MutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) raises -> None:
    _merge_cpu[dtype, width, backward=True](
        d_src_ptr, d_dst_ptr, batch_size, seq_len, num_heads, head_dim
    )


def merge_bwd[
    dtype: DType,
    target: StaticString,
](
    d_dst_ptr: ImmutKernelPtr[dtype],
    d_src_ptr: MutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises:
    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        merge_bwd_cpu[dtype, width](
            d_dst_ptr,
            d_src_ptr,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        )
    elif is_gpu[target]():
        var channels = num_heads * head_dim
        var num_elements = batch_size * num_heads * seq_len * head_dim
        comptime BLOCK_SIZE = 256
        var num_chunks = (num_elements + MERGE_GPU_WIDTH - 1) // MERGE_GPU_WIDTH
        var num_blocks = (num_chunks + BLOCK_SIZE - 1) // BLOCK_SIZE
        comptime gpu_kernel = merge_bwd_gpu[dtype, BLOCK_SIZE]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            d_dst_ptr,
            d_src_ptr,
            Int64(batch_size),
            Int64(seq_len),
            Int64(num_heads),
            Int64(head_dim),
            Int64(channels),
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


@compiler.register("merge_bwd")
struct MergeBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_src: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        d_dst: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        num_heads: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        merge_bwd[dtype, target](
            d_dst.unsafe_ptr(),
            d_src.unsafe_ptr(),
            Int(batch_size),
            Int(seq_len),
            Int(num_heads),
            Int(head_dim),
            ctx,
        )
