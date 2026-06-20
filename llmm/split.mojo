import compiler
from std.sys import simd_width_of
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.algorithm import vectorize, sync_parallelize
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from extensibility import InputTensor
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from llmm.head_layout import (
    head_layout_flat_at_head_dim_zero,
    null_immut_ptr,
    null_mut_ptr,
    token_layout_plane0_flat_at_head_dim_zero,
    token_layout_plane0_flat_from_head_layout_flat,
)

# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Split helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _split_planes[
    dtype: DType,
    width: Int,
    num_splits: Int,
    backward: Bool,
](
    qkv_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    plane0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    token_layout_src_flat_index: Int,
    head_layout_dst_flat_index: Int,
    channels: Int,
) -> None:
    comptime if backward:
        if num_splits >= 1 and Int(plane0) != 0:
            (qkv_ptr + token_layout_src_flat_index).store(
                (plane0 + head_layout_dst_flat_index).load[width=width]()
            )
        if num_splits >= 2 and Int(plane1) != 0:
            (qkv_ptr + token_layout_src_flat_index + channels).store(
                (plane1 + head_layout_dst_flat_index).load[width=width]()
            )
        if num_splits >= 3 and Int(plane2) != 0:
            (qkv_ptr + token_layout_src_flat_index + 2 * channels).store(
                (plane2 + head_layout_dst_flat_index).load[width=width]()
            )
        if num_splits >= 4 and Int(plane3) != 0:
            (qkv_ptr + token_layout_src_flat_index + 3 * channels).store(
                (plane3 + head_layout_dst_flat_index).load[width=width]()
            )
    else:
        var src_immut = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            qkv_ptr
        )
        if num_splits >= 1 and Int(dst0) != 0:
            (dst0 + head_layout_dst_flat_index).store(
                (src_immut + token_layout_src_flat_index).load[width=width]()
            )
        if num_splits >= 2 and Int(dst1) != 0:
            (dst1 + head_layout_dst_flat_index).store(
                (src_immut + token_layout_src_flat_index + channels).load[
                    width=width
                ]()
            )
        if num_splits >= 3 and Int(dst2) != 0:
            (dst2 + head_layout_dst_flat_index).store(
                (src_immut + token_layout_src_flat_index + 2 * channels).load[
                    width=width
                ]()
            )
        if num_splits >= 4 and Int(dst3) != 0:
            (dst3 + head_layout_dst_flat_index).store(
                (src_immut + token_layout_src_flat_index + 3 * channels).load[
                    width=width
                ]()
            )


@always_inline
def _split_cpu_tile[
    dtype: DType,
    width: Int,
    num_splits: Int,
    backward: Bool,
](
    qkv_mut_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    plane0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    token_layout_src_flat_at_head_dim_zero: Int,
    head_layout_dst_flat_at_head_dim_zero: Int,
    channels: Int,
    head_dim: Int,
) -> None:
    @always_inline
    def _simd[
        simd_w: Int
    ](head_dim_offset: Int) {
        qkv_mut_ptr,
        plane0,
        plane1,
        plane2,
        plane3,
        dst0,
        dst1,
        dst2,
        dst3,
        token_layout_src_flat_at_head_dim_zero,
        head_layout_dst_flat_at_head_dim_zero,
        channels,
    }:
        _split_planes[dtype, simd_w, num_splits, backward](
            qkv_mut_ptr,
            plane0,
            plane1,
            plane2,
            plane3,
            dst0,
            dst1,
            dst2,
            dst3,
            token_layout_src_flat_at_head_dim_zero + head_dim_offset,
            head_layout_dst_flat_at_head_dim_zero + head_dim_offset,
            channels,
        )

    vectorize[width, unroll_factor=UNROLL](head_dim, _simd)


@always_inline
def _split_cpu[
    dtype: DType,
    width: Int,
    num_splits: Int,
    backward: Bool,
](
    qkv_mut_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    plane0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) -> None:
    var channels = num_heads * head_dim

    @parameter
    def _worker(bh: Int):
        var b = bh // num_heads
        var h = bh % num_heads
        for t in range(seq_len):
            var token_layout_src_flat_at_head_dim_zero = (
                token_layout_plane0_flat_at_head_dim_zero(
                    b, t, h, seq_len, num_splits, channels, head_dim
                )
            )
            var head_layout_dst_flat_at_head_dim_zero = (
                head_layout_flat_at_head_dim_zero(
                    b, h, t, seq_len, num_heads, head_dim
                )
            )
            _split_cpu_tile[dtype, width, num_splits, backward](
                qkv_mut_ptr,
                plane0,
                plane1,
                plane2,
                plane3,
                dst0,
                dst1,
                dst2,
                dst3,
                token_layout_src_flat_at_head_dim_zero,
                head_layout_dst_flat_at_head_dim_zero,
                channels,
                head_dim,
            )

    sync_parallelize[_worker](batch_size * num_heads)


# ===----------------------------------------------------------------------=== #
# Split Forward
# ===----------------------------------------------------------------------=== #


def split_fwd_cpu[
    dtype: DType,
    width: Int,
    num_splits: Int,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) -> None:
    var null_immut = null_immut_ptr[dtype]()
    _split_cpu[dtype, width, num_splits, backward=False](
        src_ptr,
        null_immut,
        null_immut,
        null_immut,
        null_immut,
        dst0,
        dst1,
        dst2,
        dst3,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )


def split_gpu[
    dtype: DType,
    num_splits: Int,
    BLOCK_SIZE: Int,
    backward: Bool,
](
    qkv_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    plane0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    plane3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    var head_layout_flat_index = Int(block_idx.x * block_dim.x + thread_idx.x)
    var num_elements = Int(batch_size * num_heads * seq_len * head_dim)
    if head_layout_flat_index < num_elements:
        var token_layout_src_flat_index = (
            token_layout_plane0_flat_from_head_layout_flat(
                head_layout_flat_index,
                Int(head_dim),
                Int(seq_len),
                Int(num_heads),
                num_splits,
                Int(channels),
            )
        )
        _split_planes[dtype, 1, num_splits, backward](
            qkv_ptr,
            plane0,
            plane1,
            plane2,
            plane3,
            dst0,
            dst1,
            dst2,
            dst3,
            token_layout_src_flat_index,
            head_layout_flat_index,
            Int(channels),
        )


def split_fwd_gpu[
    dtype: DType,
    num_splits: Int,
    BLOCK_SIZE: Int,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst3: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    var null_immut = null_immut_ptr[dtype]()
    split_gpu[dtype, num_splits, BLOCK_SIZE, backward=False](
        src_ptr,
        null_immut,
        null_immut,
        null_immut,
        null_immut,
        dst0,
        dst1,
        dst2,
        dst3,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        channels,
    )


def split_bwd_gpu[
    dtype: DType,
    num_splits: Int,
    BLOCK_SIZE: Int,
](
    d_src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_dst0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch_size: Int64,
    seq_len: Int64,
    num_heads: Int64,
    head_dim: Int64,
    channels: Int64,
) -> None:
    var null_mut = null_mut_ptr[dtype]()
    split_gpu[dtype, num_splits, BLOCK_SIZE, backward=True](
        d_src_ptr,
        d_dst0,
        d_dst1,
        d_dst2,
        d_dst3,
        null_mut,
        null_mut,
        null_mut,
        null_mut,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        channels,
    )


def split_fwd[
    dtype: DType,
    target: StaticString,
    num_splits: Int,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst_ptrs: List[UnsafePointer[Scalar[dtype], MutAnyOrigin]],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises:
    var null_ptr = null_mut_ptr[dtype]()
    var dst0 = (
        dst_ptrs[0] if num_splits >= 1 and len(dst_ptrs) >= 1 else null_ptr
    )
    var dst1 = (
        dst_ptrs[1] if num_splits >= 2 and len(dst_ptrs) >= 2 else null_ptr
    )
    var dst2 = (
        dst_ptrs[2] if num_splits >= 3 and len(dst_ptrs) >= 3 else null_ptr
    )
    var dst3 = (
        dst_ptrs[3] if num_splits >= 4 and len(dst_ptrs) >= 4 else null_ptr
    )

    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        split_fwd_cpu[dtype, width, num_splits](
            src_ptr,
            dst0,
            dst1,
            dst2,
            dst3,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        )
    elif is_gpu[target]():
        var channels = num_heads * head_dim
        var num_elements = batch_size * num_heads * seq_len * head_dim
        comptime BLOCK_SIZE = 256
        var num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        comptime gpu_kernel = split_fwd_gpu[dtype, num_splits, BLOCK_SIZE]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            src_ptr,
            dst0,
            dst1,
            dst2,
            dst3,
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


@compiler.register("split_fwd")
struct SplitFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        dst0: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        dst1: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        dst2: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        src: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        num_heads: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        var dst_ptrs = List[UnsafePointer[Scalar[dtype], MutAnyOrigin]]()
        dst_ptrs.append(dst0.unsafe_ptr())
        dst_ptrs.append(dst1.unsafe_ptr())
        dst_ptrs.append(dst2.unsafe_ptr())
        split_fwd[dtype, target, num_splits=3](
            src.unsafe_ptr(),
            dst_ptrs,
            Int(batch_size),
            Int(seq_len),
            Int(num_heads),
            Int(head_dim),
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Split Backward
# ===----------------------------------------------------------------------=== #


def split_bwd_cpu[
    dtype: DType,
    width: Int,
    num_splits: Int,
](
    d_src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_dst0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_dst3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) -> None:
    var null_mut = null_mut_ptr[dtype]()
    _split_cpu[dtype, width, num_splits, backward=True](
        d_src_ptr,
        d_dst0,
        d_dst1,
        d_dst2,
        d_dst3,
        null_mut,
        null_mut,
        null_mut,
        null_mut,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )


def split_bwd[
    dtype: DType,
    target: StaticString,
    num_splits: Int,
](
    d_src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_dst_ptrs: List[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises:
    var null_ptr = null_immut_ptr[dtype]()
    var d_dst0 = (
        d_dst_ptrs[0] if num_splits >= 1 and len(d_dst_ptrs) >= 1 else null_ptr
    )
    var d_dst1 = (
        d_dst_ptrs[1] if num_splits >= 2 and len(d_dst_ptrs) >= 2 else null_ptr
    )
    var d_dst2 = (
        d_dst_ptrs[2] if num_splits >= 3 and len(d_dst_ptrs) >= 3 else null_ptr
    )
    var d_dst3 = (
        d_dst_ptrs[3] if num_splits >= 4 and len(d_dst_ptrs) >= 4 else null_ptr
    )

    comptime width = simd_width_of[dtype]()
    comptime if is_cpu[target]():
        split_bwd_cpu[dtype, width, num_splits](
            d_src_ptr,
            d_dst0,
            d_dst1,
            d_dst2,
            d_dst3,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        )
    elif is_gpu[target]():
        var channels = num_heads * head_dim
        var num_elements = batch_size * num_heads * seq_len * head_dim
        comptime BLOCK_SIZE = 256
        var num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        comptime gpu_kernel = split_bwd_gpu[dtype, num_splits, BLOCK_SIZE]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            d_src_ptr,
            d_dst0,
            d_dst1,
            d_dst2,
            d_dst3,
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


@compiler.register("split_bwd")
struct SplitBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        d_src: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        d_dst0: InputTensor[dtype=dtype, rank=1, static_spec=...],
        d_dst1: InputTensor[dtype=dtype, rank=1, static_spec=...],
        d_dst2: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        num_heads: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        var d_dst_ptrs = List[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]]()
        d_dst_ptrs.append(d_dst0.unsafe_ptr())
        d_dst_ptrs.append(d_dst1.unsafe_ptr())
        d_dst_ptrs.append(d_dst2.unsafe_ptr())
        split_bwd[dtype, target, num_splits=3](
            d_src.unsafe_ptr(),
            d_dst_ptrs,
            Int(batch_size),
            Int(seq_len),
            Int(num_heads),
            Int(head_dim),
            ctx,
        )
