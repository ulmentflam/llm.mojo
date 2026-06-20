from std.algorithm import vectorize


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
def null_mut_ptr[dtype: DType]() -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var zero = 0
    return UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=zero)


@always_inline
def null_immut_ptr[
    dtype: DType,
]() -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
    var zero = 0
    return UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=zero
    )


@always_inline
def layout_copy[
    dtype: DType,
    width: Int,
    backward: Bool,
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_flat_index: Int,
    src_flat_index: Int,
) -> None:
    comptime if backward:
        (dst_ptr + src_flat_index).store(
            (src_ptr + dst_flat_index).load[width=width]()
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
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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
