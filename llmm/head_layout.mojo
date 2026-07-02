from std.algorithm import vectorize
from std.gpu.memory import CacheOperation, load as gpu_cache_load
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# NOTE: I am trying this layout to see if it does a decent job of saving code and
# logically splitting the code into smaller pieces. I will also consider breaking out
# forward and backward passes into separate files in a similar manner.


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Head Layout Index Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def head_layout_flat_at_head_dim_zero(
    b: Int,
    h: Int,
    t: Int,
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
) -> Int:
    """Flat index of (b, h, t) with head_dim_offset = 0 in head layout."""
    return (
        b * num_heads * seq_len * head_dim
        + h * seq_len * head_dim
        + t * head_dim
    )


@always_inline
def token_layout_plane0_flat_at_head_dim_zero(
    b: Int,
    t: Int,
    h: Int,
    seq_len: Int,
    num_planes: Int,
    channels: Int,
    head_dim: Int,
) -> Int:
    """Flat index of (b, t, plane=0, h) with head_dim_offset = 0 in token layout.
    """
    return (
        b * seq_len * num_planes * channels
        + t * num_planes * channels
        + h * head_dim
    )


@always_inline
def decode_head_layout_flat_index(
    head_layout_flat_index: Int,
    head_dim: Int,
    seq_len: Int,
    num_heads: Int,
) -> Tuple[Int, Int, Int, Int]:
    """Invert head-layout flat index -> (b, h, t, head_dim_offset)."""
    var head_dim_offset = head_layout_flat_index % head_dim
    var flat_after_head_dim = head_layout_flat_index // head_dim
    var t = flat_after_head_dim % seq_len
    var flat_after_token = flat_after_head_dim // seq_len
    var h = flat_after_token % num_heads
    var b = flat_after_token // num_heads
    return (b, h, t, head_dim_offset)


@always_inline
def token_layout_plane0_flat_from_head_layout_flat(
    head_layout_flat_index: Int,
    head_dim: Int,
    seq_len: Int,
    num_heads: Int,
    num_planes: Int,
    channels: Int,
) -> Int:
    """Same element as head_layout_flat_index, indexed in token layout (plane 0).
    """
    var b, h, t, head_dim_offset = decode_head_layout_flat_index(
        head_layout_flat_index, head_dim, seq_len, num_heads
    )
    return (
        token_layout_plane0_flat_at_head_dim_zero(
            b, t, h, seq_len, num_planes, channels, head_dim
        )
        + head_dim_offset
    )


# ===----------------------------------------------------------------------=== #
# Copy helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def null_mut_ptr[dtype: DType]() -> MutKernelPtr[dtype]:
    var zero = 0
    return MutKernelPtr[dtype](unsafe_from_address=zero)


@always_inline
def null_immut_ptr[
    dtype: DType,
]() -> ImmutKernelPtr[dtype]:
    var zero = 0
    return ImmutKernelPtr[dtype](unsafe_from_address=zero)


@always_inline
def layout_copy[
    dtype: DType,
    width: Int,
    backward: Bool,
    streaming: Bool = False,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    dst_flat_index: Int,
    src_flat_index: Int,
) -> None:
    # `streaming` (item 4 experiment): llm.c's permute/unpermute use
    # __ldcs/__stcs ("this data is touched exactly once, don't pollute the
    # cache") on every load/store — this is a genuinely single-touch
    # permutation, so the streaming hint is a legitimate match. Mojo exposes
    # the load side via std.gpu.memory's `load[cache_policy=STREAMING]` and
    # the store side via UnsafePointer.store's `non_temporal=True` (maps to
    # `st.global.cs`). Both are only honored at >= 4-byte transactions (see
    # std/gpu/memory/memory.mojo's `_load_impl` width floor), so this is only
    # meaningful at width >= 2 for bf16 — at width=1 (2B) the load silently
    # falls back to a plain load, which is why width and streaming are
    # swept together, not compared at width=1.
    comptime if backward:
        comptime if streaming:
            (dst_ptr + src_flat_index).store[width=width, non_temporal=True](
                gpu_cache_load[
                    width=width, cache_policy=CacheOperation.STREAMING
                ](src_ptr + dst_flat_index)
            )
        else:
            (dst_ptr + src_flat_index).store(
                (src_ptr + dst_flat_index).load[width=width]()
            )
    else:
        comptime if streaming:
            (dst_ptr + dst_flat_index).store[width=width, non_temporal=True](
                gpu_cache_load[
                    width=width, cache_policy=CacheOperation.STREAMING
                ](src_ptr + src_flat_index)
            )
        else:
            (dst_ptr + dst_flat_index).store(
                (src_ptr + src_flat_index).load[width=width]()
            )


@always_inline
def vectorize_layout_copy[
    dtype: DType,
    width: Int,
    backward: Bool,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: ImmutKernelPtr[dtype],
    dst_flat_at_head_dim_zero: Int,
    src_flat_at_head_dim_zero: Int,
    head_dim: Int,
) -> None:
    # NOTE: This gets written fairly frequently for CPU passes.
    # We should consider making this even more generic with a wrapper function in the future.
    @always_inline
    def _simd[
        simd_w: Int
    ](
        head_dim_offset: Int,
    ) {
        dst_ptr, src_ptr, dst_flat_at_head_dim_zero, src_flat_at_head_dim_zero
    }:
        layout_copy[dtype, simd_w, backward](
            dst_ptr,
            src_ptr,
            dst_flat_at_head_dim_zero + head_dim_offset,
            src_flat_at_head_dim_zero + head_dim_offset,
        )

    vectorize[width, unroll_factor=UNROLL](head_dim, _simd)
