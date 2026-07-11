"""Unit tests for llmm/nvfp4_quant.mojo.

Run: `pixi run mojo run -I . tests/test_nvfp4_quant.mojo` (GPU tests inside
need the flock'd GPU per project policy — see AGENTS.md /
docs/ai/fp4_modular_support_research.md; the host-only numerics/packing/
swizzle tests below need no device at all).
"""

from std.math import sqrt
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true, TestSuite

from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.rand import MT19937

from _lowp_test_common import pseudo_gaussian_fill
from llmm.nvfp4_quant import (
    encode_e2m1,
    decode_e2m1,
    pack_e2m1x2,
    unpack_e2m1x2_lo,
    unpack_e2m1x2_hi,
    encode_e4m3,
    decode_e4m3,
    nvfp4_swizzled_scale_buffer_size,
    nvfp4_scale_swizzle_offset,
    nvfp4_packed_size,
    nvfp4_scale_buffer_size,
    nvfp4_quantize,
    nvfp4_quantize_transpose,
    nvfp4_dequant_reference,
)


# ===----------------------------------------------------------------------=== #
# e2m1 codec (host-only, no GPU)
# ===----------------------------------------------------------------------=== #


def test_e2m1_grid_points_roundtrip() raises:
    # Every non-negative-zero grid point (magnitude x sign) maps to itself
    # under decode -> encode. Code 8 (negative zero) is excluded: -0.0 < 0.0
    # is false in IEEE comparison, so encode(decode(8)) collapses to 0 (a
    # documented, intentional -0/+0 merge, tested separately below).
    for code in range(16):
        if code == 8:
            continue
        var x = decode_e2m1(UInt8(code))
        var back = encode_e2m1(x)
        assert_equal(Int(back), code)


def test_e2m1_negative_zero_collapses_to_positive() raises:
    var neg_zero = decode_e2m1(UInt8(8))
    assert_equal(neg_zero, Float32(0.0))
    assert_equal(Int(encode_e2m1(neg_zero)), 0)


def test_e2m1_ties_to_lower_magnitude() raises:
    # Exact midpoints between adjacent magnitudes round to the LOWER
    # magnitude (strict-`<`-only-updates tie behavior). Values: (0,0.5)->0.25,
    # (0.5,1)->0.75, (1,1.5)->1.25, (1.5,2)->1.75, (2,3)->2.5, (3,4)->3.5,
    # (4,6)->5.0.
    var ties: List[Float32] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    var lower_idx: List[Int] = [0, 1, 2, 3, 4, 5, 6]
    for i in range(len(ties)):
        var code = encode_e2m1(ties[i])
        assert_equal(Int(code) & 0x7, lower_idx[i])
        # and the negative side mirrors it (sign bit set, same magnitude idx)
        var neg_code = encode_e2m1(-ties[i])
        assert_equal(Int(neg_code) & 0x7, lower_idx[i])
        assert_true(Int(neg_code) & 0x8 != 0)


def test_e2m1_saturation() raises:
    assert_equal(Int(encode_e2m1(Float32(100.0))) & 0x7, 7)
    assert_equal(Int(encode_e2m1(Float32(-100.0))), 8 | 7)
    assert_equal(decode_e2m1(encode_e2m1(Float32(1000.0))), Float32(6.0))


def test_pack_unpack_e2m1x2_all_pairs() raises:
    for lo in range(16):
        for hi in range(16):
            var byte = pack_e2m1x2(UInt8(lo), UInt8(hi))
            assert_equal(Int(unpack_e2m1x2_lo(byte)), lo)
            assert_equal(Int(unpack_e2m1x2_hi(byte)), hi)


# ===----------------------------------------------------------------------=== #
# e4m3 codec (host-only, no GPU)
# ===----------------------------------------------------------------------=== #


def test_e4m3_zero_and_sign() raises:
    assert_equal(Int(encode_e4m3(Float32(0.0))), 0x00)
    assert_equal(decode_e4m3(UInt8(0x00)), Float32(0.0))
    # a positive-only-scale module, but sign is handled generally
    assert_equal(Int(encode_e4m3(Float32(-0.0))), 0x00)


def test_e4m3_denormal_fp32_flushes_to_zero() raises:
    # fp32 subnormal input (< ~1.18e-38), far below e4m3's own smallest
    # subnormal (2^-9 ~= 0.00195): flushes to zero, not NaN/garbage.
    var tiny = Float32(1.0e-40)
    assert_equal(Int(encode_e4m3(tiny)), 0x00)


def test_e4m3_saturation() raises:
    var code = encode_e4m3(Float32(1.0e30))
    assert_equal(decode_e4m3(code), Float32(448.0))


def test_e4m3_golden_value() raises:
    # amax=6.5 over the 16-value block in test_quantize_golden_vector below
    # -> scale = amax/6.0. Cross-checked against an independent Python
    # implementation of encode_e4m3 (frexp-based; this Mojo version is
    # bitcast-based but mathematically identical -- see module docstring).
    # Both give sc_code = 0x39 (e_biased=7, m3=1) and decode(0x39) == 1.125
    # exactly.
    var scale = Float32(6.5) / Float32(6.0)
    var code = encode_e4m3(scale)
    assert_equal(Int(code), 0x39)
    assert_equal(decode_e4m3(code), Float32(1.125))


def test_e4m3_roundtrip_relative_error_bounded() raises:
    # e4m3 has 3 mantissa bits -> worst-case relative rounding error for a
    # normal value is 1/16 (half a mantissa ULP relative to the value).
    var xs: List[Float32] = [0.01, 0.1, 1.0, 2.5, 10.0, 100.0, 447.0]
    for i in range(len(xs)):
        var x = xs[i]
        var got = decode_e4m3(encode_e4m3(x))
        var rel_err = (got - x) / x
        var abs_rel = rel_err if rel_err >= 0.0 else -rel_err
        assert_true(abs_rel <= 0.0625 + 1e-6)


# ===----------------------------------------------------------------------=== #
# Golden vector for a fixed 16-value block, independently computed in
# Python and checked against this codec's encode_e2m1/encode_e4m3/
# pack_e2m1x2 (reproduced verbatim here as literals).
# ===----------------------------------------------------------------------=== #


def test_quantize_golden_vector() raises:
    var x: List[Float32] = [
        0.1,
        -0.3,
        0.6,
        -0.9,
        1.2,
        -1.6,
        2.1,
        -2.9,
        3.4,
        -3.9,
        4.5,
        -5.5,
        5.9,
        -0.05,
        0.0,
        -6.5,
    ]
    var expected_codes: List[Int] = [
        0x0,
        0x9,
        0x1,
        0xA,
        0x2,
        0xB,
        0x4,
        0xD,
        0x5,
        0xD,
        0x6,
        0xE,
        0x7,
        0x8,
        0x0,
        0xF,
    ]
    var expected_bytes: List[Int] = [
        0x90,
        0xA1,
        0xB2,
        0xD4,
        0xD5,
        0xE6,
        0x87,
        0xF0,
    ]

    var sc_code = encode_e4m3(Float32(6.5) / Float32(6.0))
    assert_equal(Int(sc_code), 0x39)
    var sc_val = decode_e4m3(sc_code)
    assert_equal(sc_val, Float32(1.125))

    var codes = List[UInt8]()
    for i in range(16):
        var c = encode_e2m1(x[i] / sc_val)
        assert_equal(Int(c), expected_codes[i])
        codes.append(c)

    for i in range(8):
        var byte = pack_e2m1x2(codes[2 * i], codes[2 * i + 1])
        assert_equal(Int(byte), expected_bytes[i])


# ===----------------------------------------------------------------------=== #
# Scale swizzle (host-only, no GPU)
# ===----------------------------------------------------------------------=== #


def test_swizzle_golden_offsets() raises:
    # rows=20, cols=9 -> n_col_tiles = ceildiv(9,4) = 3, buffer = 32*1*16*3=1536
    var n_col_tiles = 3
    assert_equal(nvfp4_swizzled_scale_buffer_size(20, 9), 1536)
    assert_equal(nvfp4_scale_swizzle_offset(0, 0, n_col_tiles), 0)
    assert_equal(nvfp4_scale_swizzle_offset(5, 1, n_col_tiles), 81)
    assert_equal(nvfp4_scale_swizzle_offset(19, 8, n_col_tiles), 1328)
    assert_equal(nvfp4_scale_swizzle_offset(0, 4, n_col_tiles), 512)
    assert_equal(nvfp4_scale_swizzle_offset(3, 7, n_col_tiles), 563)


def test_swizzle_bijection_no_collisions() raises:
    # Every logical (row, col) must map to a distinct byte offset, all
    # within the buffer, across a shape spanning multiple row/col tiles
    # (128-row and 4-col boundaries).
    var rows = 140
    var cols = 10
    var n_col_tiles = (cols + 3) // 4
    var buf_size = nvfp4_swizzled_scale_buffer_size(rows, cols)
    var seen = List[Bool]()
    for _ in range(buf_size):
        seen.append(False)
    for r in range(rows):
        for c in range(cols):
            var off = nvfp4_scale_swizzle_offset(r, c, n_col_tiles)
            assert_true(off >= 0 and off < buf_size)
            assert_true(not seen[off])
            seen[off] = True


# ===----------------------------------------------------------------------=== #
# GPU: quantize kernel round-trip / ill-conditioned inputs
# ===----------------------------------------------------------------------=== #


def _quantize_roundtrip_case[
    BLOCK_ROWS: Int
](
    rows: Int, k: Int, mean: Float32, std: Float32, seed: UInt32
) raises -> Float64:
    """Generates deterministic gaussian data, quantizes on GPU with the
    given BLOCK_ROWS granularity, dequantizes via the host reference, and
    returns the relative-L2 error vs the original fp32 data.
    """
    var ctx = DeviceContext()
    var n = rows * k

    var host_fp32 = ctx.enqueue_create_host_buffer[DType.float32](n)
    var rng = MT19937(seed)
    pseudo_gaussian_fill(rng, host_fp32.unsafe_ptr(), n, std, mean)
    ctx.synchronize()

    var host_bf16 = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host_bf16.unsafe_ptr()[i] = host_fp32.unsafe_ptr()[i].cast[
            DType.bfloat16
        ]()

    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_bf16)

    var q_size = nvfp4_packed_size(rows, k)
    var scale_size = nvfp4_scale_buffer_size(rows, k, BLOCK_ROWS)
    var q_dev = ctx.enqueue_create_buffer[DType.uint8](q_size)
    var scale_dev = ctx.enqueue_create_buffer[DType.uint8](scale_size)
    var tensor_scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    nvfp4_quantize[DType.bfloat16, "gpu", BLOCK_ROWS](
        rebind[MutKernelPtr[DType.uint8]](
            q_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            tensor_scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_q = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
    var host_scale = ctx.enqueue_create_host_buffer[DType.uint8](scale_size)
    var host_tensor_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    q_dev.enqueue_copy_to(host_q)
    scale_dev.enqueue_copy_to(host_scale)
    tensor_scale_dev.enqueue_copy_to(host_tensor_scale)
    ctx.synchronize()

    var host_recon = ctx.enqueue_create_host_buffer[DType.float32](n)
    nvfp4_dequant_reference[BLOCK_ROWS](
        host_recon.unsafe_ptr(),
        host_q.unsafe_ptr().as_immutable(),
        host_scale.unsafe_ptr().as_immutable(),
        host_tensor_scale.unsafe_ptr()[0],
        rows,
        k,
    )

    var sq_err = Float64(0.0)
    var sq_ref = Float64(0.0)
    for i in range(n):
        var orig = Float64(host_fp32.unsafe_ptr()[i])
        var got = Float64(host_recon.unsafe_ptr()[i])
        var e = got - orig
        sq_err += e * e
        sq_ref += orig * orig

    return sqrt(sq_err / (sq_ref if sq_ref > 0.0 else 1.0))


def test_quantize_1d_roundtrip_gaussian_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var rel_l2 = _quantize_roundtrip_case[1](
        32, 256, Float32(0.0), Float32(1.0), UInt32(42)
    )
    assert_true(rel_l2 < 0.20)


def test_quantize_2d_roundtrip_gaussian_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var rel_l2 = _quantize_roundtrip_case[16](
        64, 256, Float32(0.0), Float32(1.0), UInt32(43)
    )
    # Coarser granularity (256 elements/scale vs 16) -> looser bound.
    assert_true(rel_l2 < 0.35)


def test_quantize_transpose_matches_materialized_transpose_gpu() raises:
    # `nvfp4_quantize_transpose` (backward) must produce
    # BYTE-IDENTICAL output to materializing the transpose in bf16 host-side
    # and calling plain `nvfp4_quantize` on it -- both are the SAME
    # deterministic RNE computation over the SAME set of logical values, just
    # reached via a different physical memory-read pattern (module docstring
    # in llmm/nvfp4_quant.mojo, "Transposed quantize" section). A bug in the
    # `_nvfp4_src_addr` index-swap would show up here as a byte mismatch,
    # not merely a "somewhat higher error" -- a much sharper signal than the
    # end-to-end Dgrad/Wgrad GEMM tests (tests/test_matmul_bwd_fp4.mojo),
    # which could not distinguish a subtly-wrong transpose from ordinary fp4
    # quantization noise.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime BLOCK_ROWS = 1
    # source [src_rows, src_k]; logical (transposed) output [src_k, src_rows]
    # -- src_rows must be a multiple of 16 (nvfp4_quantize_transpose's
    # docstring: it becomes the logical output's own block-16 axis).
    var src_rows = 32
    var src_k = 48
    var n = src_rows * src_k

    var rng = MT19937(UInt32(777))
    var host_src_fp32 = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        var s = Float32(0.0)
        for _ in range(12):
            s += rng.randfloat32()
        host_src_fp32.unsafe_ptr()[i] = (s - 6.0) * Float32(1.0)
    var host_src_bf16 = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host_src_bf16.unsafe_ptr()[i] = host_src_fp32.unsafe_ptr()[i].cast[
            DType.bfloat16
        ]()

    # Materialize the transpose host-side: T[i,j] = src[j,i], T is
    # [src_k, src_rows] row-major.
    var host_t_bf16 = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(src_k):
        for j in range(src_rows):
            host_t_bf16.unsafe_ptr()[
                i * src_rows + j
            ] = host_src_bf16.unsafe_ptr()[j * src_k + i]

    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_src_bf16)
    var t_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    t_dev.enqueue_copy_from(host_t_bf16)

    var q_size = nvfp4_packed_size(src_k, src_rows)
    var scale_size = nvfp4_scale_buffer_size(src_k, src_rows, BLOCK_ROWS)
    var q_transpose = ctx.enqueue_create_buffer[DType.uint8](q_size)
    var scale_transpose = ctx.enqueue_create_buffer[DType.uint8](scale_size)
    var tscale_transpose = ctx.enqueue_create_buffer[DType.float32](1)
    var q_ref = ctx.enqueue_create_buffer[DType.uint8](q_size)
    var scale_ref = ctx.enqueue_create_buffer[DType.uint8](scale_size)
    var tscale_ref = ctx.enqueue_create_buffer[DType.float32](1)
    # The swizzled scale buffer has PADDING (128-row / 4-col tile geometry,
    # 512 bytes total here vs 96 actually-written entries) -- zero both
    # buffers first so never-written padding bytes compare equal (0 == 0)
    # instead of comparing two independently-allocated buffers' leftover
    # garbage, which would fail this test for a reason that has nothing to
    # do with `nvfp4_quantize_transpose`'s correctness.
    ctx.enqueue_memset(scale_transpose, Scalar[DType.uint8](0))
    ctx.enqueue_memset(scale_ref, Scalar[DType.uint8](0))
    ctx.synchronize()

    nvfp4_quantize_transpose[DType.bfloat16, "gpu", BLOCK_ROWS](
        rebind[MutKernelPtr[DType.uint8]](
            q_transpose.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            scale_transpose.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            tscale_transpose.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        src_rows,
        src_k,
        ctx,
    )
    nvfp4_quantize[DType.bfloat16, "gpu", BLOCK_ROWS](
        rebind[MutKernelPtr[DType.uint8]](
            q_ref.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            scale_ref.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            tscale_ref.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            t_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        src_k,
        src_rows,
        ctx,
    )
    ctx.synchronize()

    var host_q_transpose = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
    var host_q_ref = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
    var host_scale_transpose = ctx.enqueue_create_host_buffer[DType.uint8](
        scale_size
    )
    var host_scale_ref = ctx.enqueue_create_host_buffer[DType.uint8](scale_size)
    var host_tscale_transpose = ctx.enqueue_create_host_buffer[DType.float32](1)
    var host_tscale_ref = ctx.enqueue_create_host_buffer[DType.float32](1)
    q_transpose.enqueue_copy_to(host_q_transpose)
    q_ref.enqueue_copy_to(host_q_ref)
    scale_transpose.enqueue_copy_to(host_scale_transpose)
    scale_ref.enqueue_copy_to(host_scale_ref)
    tscale_transpose.enqueue_copy_to(host_tscale_transpose)
    tscale_ref.enqueue_copy_to(host_tscale_ref)
    ctx.synchronize()

    assert_equal(
        host_tscale_transpose.unsafe_ptr()[0], host_tscale_ref.unsafe_ptr()[0]
    )
    for i in range(q_size):
        assert_equal(
            host_q_transpose.unsafe_ptr()[i],
            host_q_ref.unsafe_ptr()[i],
            "packed e2m1 byte mismatch at " + String(i),
        )
    for i in range(scale_size):
        assert_equal(
            host_scale_transpose.unsafe_ptr()[i],
            host_scale_ref.unsafe_ptr()[i],
            "swizzled e4m3 scale byte mismatch at " + String(i),
        )


def test_quantize_all_zero_tensor_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var rows = 8
    var k = 32
    var n = rows * k

    var host_bf16 = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host_bf16.unsafe_ptr()[i] = Float32(0.0).cast[DType.bfloat16]()
    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_bf16)

    var q_size = nvfp4_packed_size(rows, k)
    var scale_size = nvfp4_scale_buffer_size(rows, k, 1)
    var q_dev = ctx.enqueue_create_buffer[DType.uint8](q_size)
    var scale_dev = ctx.enqueue_create_buffer[DType.uint8](scale_size)
    var tensor_scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    nvfp4_quantize[DType.bfloat16, "gpu", 1](
        rebind[MutKernelPtr[DType.uint8]](
            q_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            tensor_scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_tensor_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    tensor_scale_dev.enqueue_copy_to(host_tensor_scale)
    var host_q = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
    q_dev.enqueue_copy_to(host_q)
    ctx.synchronize()

    # No div-by-zero/NaN fallback: tensor_scale falls back to 1.0.
    assert_equal(host_tensor_scale.unsafe_ptr()[0], Float32(1.0))
    for i in range(q_size):
        var byte = host_q.unsafe_ptr()[i]
        assert_equal(Int(unpack_e2m1x2_lo(byte)) & 0x7, 0)
        assert_equal(Int(unpack_e2m1x2_hi(byte)) & 0x7, 0)


def test_quantize_large_outlier_no_overflow_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var rows = 4
    var k = 32
    var n = rows * k

    var host_fp32 = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host_fp32.unsafe_ptr()[i] = 0.001
    # One extreme outlier drives a huge block amax.
    host_fp32.unsafe_ptr()[0] = 1.0e6

    var host_bf16 = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host_bf16.unsafe_ptr()[i] = host_fp32.unsafe_ptr()[i].cast[
            DType.bfloat16
        ]()
    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_bf16)

    var q_size = nvfp4_packed_size(rows, k)
    var scale_size = nvfp4_scale_buffer_size(rows, k, 1)
    var q_dev = ctx.enqueue_create_buffer[DType.uint8](q_size)
    var scale_dev = ctx.enqueue_create_buffer[DType.uint8](scale_size)
    var tensor_scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    nvfp4_quantize[DType.bfloat16, "gpu", 1](
        rebind[MutKernelPtr[DType.uint8]](
            q_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            tensor_scale_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_recon = ctx.enqueue_create_host_buffer[DType.float32](n)
    var host_q = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
    var host_scale = ctx.enqueue_create_host_buffer[DType.uint8](scale_size)
    var host_tensor_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    q_dev.enqueue_copy_to(host_q)
    scale_dev.enqueue_copy_to(host_scale)
    tensor_scale_dev.enqueue_copy_to(host_tensor_scale)
    ctx.synchronize()

    nvfp4_dequant_reference[1](
        host_recon.unsafe_ptr(),
        host_q.unsafe_ptr().as_immutable(),
        host_scale.unsafe_ptr().as_immutable(),
        host_tensor_scale.unsafe_ptr()[0],
        rows,
        k,
    )

    for i in range(n):
        var v = host_recon.unsafe_ptr()[i]
        assert_true(v == v)  # not NaN
        assert_true(v > -1.0e8 and v < 1.0e8)  # not +/-inf-ish


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
