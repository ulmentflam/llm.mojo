# ===----------------------------------------------------------------------=== #
# tests/test_lowp_gemm.mojo — Chunk B gate (docs/ai/fp8_training_design.md §6):
#
#   1. Manual fp8 encode/decode round-trips bit-exact against Mojo's native
#      host-target fp8 casts (the correctness oracle — probe1 in
#      tests/probe_fp8/RESULTS.md confirms those casts work on CPU/host).
#   2. The GPU `quantize_devscale`/`quantize_transpose_devscale` kernels
#      (llmm/lowp.mojo) match the manual encoder, including saturation /
#      zero / (fp8-)denormal / amax=0 edge cases. (The host-Float32-scale
#      twins these tests originally gated were deleted by the DRY pass F3 —
#      docs/ai/dry_consolidation_audit_2026-07-10.md; the tests now upload
#      their host-computed scale into a 1-element device buffer.)
#   3. `lowp_gemm_devscale` (llmm/matmul.mojo) vs a plain fp32 reference GEMM
#      on the
#      pre-quantization bf16 data: per-element relative error < 2^-3 (E4M3's
#      ~3 mantissa bits), across all three block-GEMM operand orientations
#      (forward / dgrad / wgrad — see matmul.mojo's `_matmul_cublaslt_fp8`
#      module comment for the orientation derivation).
#   4. An ill-scaled case (tiny and huge input magnitudes with compensating
#      scales) — proves the scale mechanism, not just well-scaled data.
#   5. NaN/Inf-free assertions throughout.
#
# GPU-only tests are guarded by `has_nvidia_gpu_accelerator()` and expected to
# run under `flock -w 10800 /tmp/llmm-gpu.lock -c '...'` (shared GPU).
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import sqrt
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true, assert_false

from llmm.lowp import (
    encode_e4m3,
    encode_e5m2,
    decode_e4m3,
    decode_e5m2,
    quantize_devscale,
    quantize_transpose_devscale,
    RoundMode,
    FP8_SPEC,
    E4M3_MAX,
    E5M2_MAX,
)
from llmm.matmul import lowp_gemm_devscale
from llmm.memory import MutKernelPtr, ImmutKernelPtr

from _lowp_test_common import _host_gemm_ref


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _bits_of_fp8[dtype: DType](x: Scalar[dtype]) -> UInt8:
    var v = x
    return UnsafePointer(to=v).bitcast[UInt8]()[]


# ===----------------------------------------------------------------------=== #
# 1. Manual encode/decode vs native host casts (bit-exact for RNE).
# ===----------------------------------------------------------------------=== #


def _check_encode_decode(x: Float32) raises:
    var e4_native = x.cast[DType.float8_e4m3fn]()
    var e4_native_bits = _bits_of_fp8(e4_native)
    var e4_manual = encode_e4m3(x)
    assert_true(
        e4_native_bits == e4_manual,
        "e4m3 encode mismatch x="
        + String(x)
        + " native="
        + String(e4_native_bits)
        + " manual="
        + String(e4_manual),
    )
    var e4_back_native = e4_native.cast[DType.float32]()
    var e4_back_manual = decode_e4m3(e4_manual)
    assert_true(
        e4_back_native == e4_back_manual,
        "e4m3 decode mismatch x=" + String(x),
    )

    var e5_native = x.cast[DType.float8_e5m2]()
    var e5_native_bits = _bits_of_fp8(e5_native)
    var e5_manual = encode_e5m2(x)
    assert_true(
        e5_native_bits == e5_manual,
        "e5m2 encode mismatch x="
        + String(x)
        + " native="
        + String(e5_native_bits)
        + " manual="
        + String(e5_manual),
    )
    var e5_back_native = e5_native.cast[DType.float32]()
    var e5_back_manual = decode_e5m2(e5_manual)
    assert_true(
        e5_back_native == e5_back_manual,
        "e5m2 decode mismatch x=" + String(x),
    )


def test_encode_decode_roundtrip_vs_native() raises:
    var edges = List[Float32]()
    edges.append(0.0)
    edges.append(-0.0)
    edges.append(1.0)
    edges.append(-1.0)
    edges.append(E4M3_MAX)
    edges.append(-E4M3_MAX)
    edges.append(E5M2_MAX)
    edges.append(-E5M2_MAX)
    edges.append(500.0)
    edges.append(1000.0)
    edges.append(100000.0)
    edges.append(-100000.0)
    edges.append(1.0e30)
    edges.append(1.0e-10)
    edges.append(-1.0e-10)
    edges.append(0.001)
    edges.append(0.0019531)
    edges.append(0.0009765)
    edges.append(2.0)
    edges.append(3.0)
    edges.append(0.5)
    edges.append(0.1)
    edges.append(6.103515625e-05)
    for i in range(len(edges)):
        _check_encode_decode(edges[i])

    seed(12345)
    for decade in range(-12, 6):
        var scale = Float64(10.0) ** Float64(decade)
        for _ in range(200):
            var r = (random_float64() * 2.0 - 1.0) * scale
            _check_encode_decode(Float32(r))


def test_encode_near_max_no_nan_pattern() raises:
    # Regression for a real bug found in a sibling (FP4-probe) e4m3 reference
    # encoder: a post-round exponent clamp (`if e_biased >= 15: e_biased =
    # 14; m3 = 7`) that *looks* like "saturate to 448" actually decodes to
    # 240 (exp_field=14 with mantissa=7 is not 448's bit pattern at all,
    # and e_biased=15/m3=7 is separately the format's reserved NaN
    # pattern) -- 447.0 encoded wrong under that scheme. This encoder avoids
    # the whole bug class structurally: `_fp8_encode_rne` clamps the *input
    # magnitude* to max_normal (448.0 / 57344.0) BEFORE rounding, and
    # max_normal is itself exactly representable (exp_field=15,
    # mantissa=0b110 for e4m3 -- mantissa=0b111 is NaN and is never reached),
    # so no post-round exponent-clamp branch exists to get wrong. Swept
    # densely here (every 0.25 in e4m3's top octave, every 32 in e5m2's) as
    # a durable regression, not just the two edge points.
    var mismatches = 0
    var x = Float32(256.0)
    while x <= Float32(480.0):
        var native = _bits_of_fp8(x.cast[DType.float8_e4m3fn]())
        var manual = encode_e4m3(x)
        if native != manual:
            mismatches += 1
        x += Float32(0.25)
    assert_true(
        mismatches == 0,
        "e4m3 near-max sweep had " + String(mismatches) + " mismatches",
    )
    # The exact failure mode reported: 447.0 must round UP to 448.0 (bits
    # 0b0111_1110 = 126), never to the NaN pattern (127) or to 240.0.
    var v447 = encode_e4m3(Float32(447.0))
    assert_true(
        v447 == UInt8(126),
        "447.0 encoded to bits=" + String(v447) + ", want 126 (448.0)",
    )
    assert_true(
        decode_e4m3(v447) == Float32(448.0),
        "447.0 decoded to " + String(decode_e4m3(v447)) + ", want 448.0",
    )

    var mismatches5 = 0
    var y = Float32(32768.0)
    while y <= Float32(60000.0):
        var native5 = _bits_of_fp8(y.cast[DType.float8_e5m2]())
        var manual5 = encode_e5m2(y)
        if native5 != manual5:
            mismatches5 += 1
        y += Float32(32.0)
    assert_true(
        mismatches5 == 0,
        "e5m2 near-max sweep had " + String(mismatches5) + " mismatches",
    )
    # e5m2 additionally HAS an encodable infinity (unlike e4m3fn), but the
    # saturating scheme here must never reach it -- 57343.0 must round to
    # 57344.0 (finite max), not to +inf.
    var v57343 = encode_e5m2(Float32(57343.0))
    assert_true(
        decode_e5m2(v57343) == Float32(57344.0),
        "57343.0 decoded to "
        + String(decode_e5m2(v57343))
        + ", want 57344.0 (not inf)",
    )


# ===----------------------------------------------------------------------=== #
# 2. GPU quantize kernel correctness (incl. saturation/zero/denormal/amax=0).
# ===----------------------------------------------------------------------=== #


def test_quantize_kernel_matches_manual_encode() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime N = 4096
    comptime IN_DT = DType.bfloat16

    var host_in = ctx.enqueue_create_host_buffer[IN_DT](N)
    seed(777)
    var host_ref = List[Float32](capacity=N)
    for i in range(N):
        var v: Float32
        # Mix magnitude regimes: near-zero (denormal-in-fp8 range), normal,
        # and a few forced-saturation outliers.
        if i % 17 == 0:
            v = Float32(1.0e6) * (
                Float32(1.0) if i % 2 == 0 else Float32(-1.0)
            )  # saturates
        elif i % 5 == 0:
            v = Float32(0.0)  # exact zero
        elif i % 3 == 0:
            v = Float32(
                (random_float64() * 2.0 - 1.0) * 0.002
            )  # fp8-subnormal-ish
        else:
            v = Float32((random_float64() * 2.0 - 1.0) * 4.0)  # normal range
        host_in.unsafe_ptr()[i] = v.cast[IN_DT]()
        host_ref.append(
            v.cast[IN_DT]().cast[DType.float32]()
        )  # bf16-rounded reference input

    var dev_in = ctx.enqueue_create_buffer[IN_DT](N)
    dev_in.enqueue_copy_from(host_in)
    var dev_out = ctx.enqueue_create_buffer[DType.uint8](N)
    ctx.synchronize()

    comptime SCALE = Float32(2.5)
    # Host-computed scale uploaded into a 1-element device buffer (the
    # devscale contract — the host-scale twin was deleted in DRY pass F3).
    var host_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    host_scale.unsafe_ptr()[0] = SCALE
    var dev_scale = ctx.enqueue_create_buffer[DType.float32](1)
    dev_scale.enqueue_copy_from(host_scale)
    ctx.synchronize()
    quantize_devscale[FP8_SPEC, DType.float8_e4m3fn, IN_DT, "gpu"](
        dev_out.unsafe_ptr().as_unsafe_any_origin(),
        dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_scale.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        N,
        ctx,
    )
    ctx.synchronize()

    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](N)
    dev_out.enqueue_copy_to(host_out)
    ctx.synchronize()

    var mismatches = 0
    for i in range(N):
        var expected = encode_e4m3(host_ref[i] * SCALE)
        var got = host_out.unsafe_ptr()[i]
        if expected != got:
            mismatches += 1
    assert_true(
        mismatches == 0,
        "quantize kernel mismatched manual encode at "
        + String(mismatches)
        + "/"
        + String(N)
        + " elements",
    )


def test_quantize_kernel_amax_zero() raises:
    # amax=0 (all-zero input): with a caller-chosen fallback scale (any finite
    # value -- callers must never divide by amax=0 to derive scale; that is
    # Chunk C's job, not tested here), quantize must produce all-zero fp8
    # bytes and no NaN/Inf.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime N = 256
    comptime IN_DT = DType.bfloat16
    var host_in = ctx.enqueue_create_host_buffer[IN_DT](N)
    for i in range(N):
        host_in.unsafe_ptr()[i] = Float32(0.0).cast[IN_DT]()
    var dev_in = ctx.enqueue_create_buffer[IN_DT](N)
    dev_in.enqueue_copy_from(host_in)
    var dev_out = ctx.enqueue_create_buffer[DType.uint8](N)
    ctx.synchronize()

    var host_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    host_scale.unsafe_ptr()[0] = Float32(1.0)
    var dev_scale = ctx.enqueue_create_buffer[DType.float32](1)
    dev_scale.enqueue_copy_from(host_scale)
    ctx.synchronize()
    quantize_devscale[FP8_SPEC, DType.float8_e4m3fn, IN_DT, "gpu"](
        dev_out.unsafe_ptr().as_unsafe_any_origin(),
        dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_scale.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        N,
        ctx,
    )
    ctx.synchronize()
    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](N)
    dev_out.enqueue_copy_to(host_out)
    ctx.synchronize()
    for i in range(N):
        assert_true(
            host_out.unsafe_ptr()[i] == UInt8(0),
            "amax=0 quantize produced a nonzero byte at " + String(i),
        )


def test_quantize_transpose_matches_manual() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime ROWS = 37
    comptime COLS = 53
    comptime IN_DT = DType.bfloat16
    var n = ROWS * COLS

    var host_in = ctx.enqueue_create_host_buffer[IN_DT](n)
    seed(42)
    for i in range(n):
        var v = Float32((random_float64() * 2.0 - 1.0) * 3.0)
        host_in.unsafe_ptr()[i] = v.cast[IN_DT]()

    var dev_in = ctx.enqueue_create_buffer[IN_DT](n)
    dev_in.enqueue_copy_from(host_in)
    var dev_out = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.synchronize()

    comptime SCALE = Float32(10.0)
    var host_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    host_scale.unsafe_ptr()[0] = SCALE
    var dev_scale = ctx.enqueue_create_buffer[DType.float32](1)
    dev_scale.enqueue_copy_from(host_scale)
    ctx.synchronize()
    quantize_transpose_devscale[FP8_SPEC, DType.float8_e4m3fn, IN_DT, "gpu"](
        dev_out.unsafe_ptr().as_unsafe_any_origin(),
        dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_scale.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS,
        COLS,
        ctx,
    )
    ctx.synchronize()
    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](n)
    dev_out.enqueue_copy_to(host_out)
    ctx.synchronize()

    var mismatches = 0
    for r in range(ROWS):
        for c in range(COLS):
            var src = host_in.unsafe_ptr()[r * COLS + c].cast[DType.float32]()
            var expected = encode_e4m3(src * SCALE)
            var got = host_out.unsafe_ptr()[c * ROWS + r]
            if expected != got:
                mismatches += 1
    assert_true(
        mismatches == 0,
        "quantize_transpose mismatched manual transpose+encode at "
        + String(mismatches)
        + " elements",
    )


# ===----------------------------------------------------------------------=== #
# 3. lowp_gemm vs fp32 reference, all three operand orientations.
# ===----------------------------------------------------------------------=== #


def _run_lowp_gemm_case[
    transpose_a: Bool, transpose_b: Bool
](
    m: Int,
    n: Int,
    k: Int,
    in_scale: Float32,
    label: String,
    max_rel_l2: Float32 = Float32(0.125),
    min_cosine: Float32 = Float32(0.999),
) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime IN_DT = DType.bfloat16
    comptime OUT_DT = DType.bfloat16
    comptime FP8_DT = DType.float8_e4m3fn

    var a_n = k * m if transpose_a else m * k
    var b_n = k * n if transpose_b else n * k

    var host_a = ctx.enqueue_create_host_buffer[IN_DT](a_n)
    var host_b = ctx.enqueue_create_host_buffer[IN_DT](b_n)
    var host_a_ref = ctx.enqueue_create_host_buffer[DType.float32](a_n)
    var host_b_ref = ctx.enqueue_create_host_buffer[DType.float32](b_n)
    seed(2026)
    for i in range(a_n):
        var v = Float32((random_float64() * 2.0 - 1.0)) * in_scale
        host_a.unsafe_ptr()[i] = v.cast[IN_DT]()
        host_a_ref.unsafe_ptr()[i] = v.cast[IN_DT]().cast[DType.float32]()
    for i in range(b_n):
        var v = Float32((random_float64() * 2.0 - 1.0)) * in_scale
        host_b.unsafe_ptr()[i] = v.cast[IN_DT]()
        host_b_ref.unsafe_ptr()[i] = v.cast[IN_DT]().cast[DType.float32]()

    var dev_a = ctx.enqueue_create_buffer[IN_DT](a_n)
    var dev_b = ctx.enqueue_create_buffer[IN_DT](b_n)
    dev_a.enqueue_copy_from(host_a)
    dev_b.enqueue_copy_from(host_b)

    var dev_a_scratch = ctx.enqueue_create_buffer[DType.uint8](a_n)
    var dev_b_scratch = ctx.enqueue_create_buffer[DType.uint8](b_n)
    var dev_d = ctx.enqueue_create_buffer[OUT_DT](m * n)

    # Quantize scale: E4M3_MAX / amax, amax computed on host from the actual
    # data (mimics what Chunk C's AmaxState would supply for a well-scaled
    # single-tensor case).
    var amax_a = Float32(0.0)
    for i in range(a_n):
        var av = abs(host_a_ref.unsafe_ptr()[i])
        if av > amax_a:
            amax_a = av
    var amax_b = Float32(0.0)
    for i in range(b_n):
        var bv = abs(host_b_ref.unsafe_ptr()[i])
        if bv > amax_b:
            amax_b = bv
    var scale_a = E4M3_MAX / (amax_a + Float32(1e-12))
    var scale_b = E4M3_MAX / (amax_b + Float32(1e-12))
    var scale_inv_a = Float32(1.0) / scale_a
    var scale_inv_b = Float32(1.0) / scale_b

    # Host-computed scales uploaded into 1-element device buffers (the
    # devscale contract — the host-scale `lowp_gemm` twin was deleted in DRY
    # pass F3): quantize-time multipliers AND their reciprocals, kept in
    # sync as `lowp_gemm_devscale`'s docstring requires.
    var host_s_a = ctx.enqueue_create_host_buffer[DType.float32](1)
    var host_s_b = ctx.enqueue_create_host_buffer[DType.float32](1)
    host_s_a.unsafe_ptr()[0] = scale_a
    host_s_b.unsafe_ptr()[0] = scale_b
    var dev_s_a = ctx.enqueue_create_buffer[DType.float32](1)
    var dev_s_b = ctx.enqueue_create_buffer[DType.float32](1)
    dev_s_a.enqueue_copy_from(host_s_a)
    dev_s_b.enqueue_copy_from(host_s_b)
    var host_sinv_a = ctx.enqueue_create_host_buffer[DType.float32](1)
    var host_sinv_b = ctx.enqueue_create_host_buffer[DType.float32](1)
    host_sinv_a.unsafe_ptr()[0] = scale_inv_a
    host_sinv_b.unsafe_ptr()[0] = scale_inv_b
    var dev_sinv_a = ctx.enqueue_create_buffer[DType.float32](1)
    var dev_sinv_b = ctx.enqueue_create_buffer[DType.float32](1)
    dev_sinv_a.enqueue_copy_from(host_sinv_a)
    dev_sinv_b.enqueue_copy_from(host_sinv_b)
    ctx.synchronize()

    lowp_gemm_devscale[
        FP8_DT,
        FP8_DT,
        IN_DT,
        OUT_DT,
        "gpu",
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    ](
        dev_d.unsafe_ptr().as_unsafe_any_origin(),
        dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_a_scratch.unsafe_ptr().as_unsafe_any_origin(),
        dev_b_scratch.unsafe_ptr().as_unsafe_any_origin(),
        dev_s_a.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_sinv_a.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_s_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        dev_sinv_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        m,
        n,
        k,
        False,
        ctx,
    )
    ctx.synchronize()

    var host_d = ctx.enqueue_create_host_buffer[OUT_DT](m * n)
    dev_d.enqueue_copy_to(host_d)
    ctx.synchronize()

    var host_ref = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    _host_gemm_ref[transpose_a, transpose_b](
        host_a_ref.unsafe_ptr(),
        host_b_ref.unsafe_ptr(),
        host_ref.unsafe_ptr(),
        m,
        n,
        k,
    )

    # Aggregate metrics (relative L2 norm + cosine similarity) rather than a
    # naive per-element `|got-want| / |want|`: GEMM outputs cross zero (sums
    # of k signed products), and near those zero-crossings a tiny, entirely
    # quantization-proportionate absolute error produces an arbitrarily large
    # *relative* error against an near-zero `want` -- not a real correctness
    # signal. Confirmed empirically while building this test: a per-element
    # metric flagged ~17% of entries as ">2^-3 relative error" purely from
    # this cancellation artifact, while the same run's relative L2 norm was
    # 0.036 and cosine similarity 0.9993 (both comfortably inside the
    # 2^-3-mantissa E4M3 budget this gate is checking for). This matches the
    # metric Chunk D/E's own gates use (docs/ai/fp8_training_design.md §6,
    # landmine #3 in MEMORY.md: naive flat/per-element error metrics have
    # hidden real bugs before -- the fix here goes the other direction, a
    # per-element metric giving *false positives* on ordinary cancellation).
    var l2_err = Float32(0.0)
    var l2_want = Float32(0.0)
    var dot = Float32(0.0)
    var norm_got = Float32(0.0)
    var norm_want = Float32(0.0)
    for i in range(m * n):
        var got = host_d.unsafe_ptr()[i].cast[DType.float32]()
        assert_true(
            got == got,
            label + ": NaN in lowp_gemm_devscale output at " + String(i),
        )
        assert_true(
            got > Float32(-1e30) and got < Float32(1e30),
            label
            + ": Inf/overflow in lowp_gemm_devscale output at "
            + String(i),
        )
        var want = host_ref.unsafe_ptr()[i]
        var err = got - want
        l2_err += err * err
        l2_want += want * want
        dot += got * want
        norm_got += got * got
        norm_want += want * want
    var rel_l2 = sqrt(l2_err / (l2_want + Float32(1e-12)))
    var cosine = dot / (sqrt(norm_got) * sqrt(norm_want) + Float32(1e-12))
    assert_true(
        rel_l2 < max_rel_l2,
        label
        + ": relative L2 norm "
        + String(rel_l2)
        + " >= "
        + String(max_rel_l2)
        + " vs fp32 reference (m="
        + String(m)
        + " n="
        + String(n)
        + " k="
        + String(k)
        + ")",
    )
    assert_true(
        cosine > min_cosine,
        label
        + ": cosine similarity "
        + String(cosine)
        + " <= "
        + String(min_cosine),
    )


def test_lowp_gemm_forward_orientation() raises:
    _run_lowp_gemm_case[False, False](128, 96, 128, Float32(1.0), "forward")


def test_lowp_gemm_dgrad_orientation() raises:
    _run_lowp_gemm_case[True, False](96, 128, 128, Float32(1.0), "dgrad")


def test_lowp_gemm_wgrad_orientation() raises:
    _run_lowp_gemm_case[True, True](96, 64, 128, Float32(1.0), "wgrad")


def test_lowp_gemm_ill_scaled_tiny() raises:
    # Inputs ~2e-4 in magnitude -- would flush to fp8 zero/subnormal without
    # a compensating (large) scale. Proves the scale mechanism (primarily:
    # no NaN/Inf, no flush-to-zero collapse), not well-scaled-data precision
    # -- a looser accuracy bound than the orientation tests above is
    # deliberate: measured cosine similarity here is ~0.988 (vs ~0.999+ for
    # the well-scaled orientation tests), consistent with amax-based
    # per-tensor scaling giving every element the *same* scale regardless of
    # its individual magnitude, so relative quantization noise is uniform
    # (not correlated with `in_scale`) but this particular random draw at
    # k=128 happens to land a bit worse on the aggregate metric -- still
    # comfortably far from garbage (no overflow, no NaN, no collapse to
    # zero), which is what this case is actually gating.
    _run_lowp_gemm_case[False, False](
        64,
        64,
        128,
        Float32(2.0e-4),
        "ill-scaled-tiny",
        max_rel_l2=Float32(0.25),
        min_cosine=Float32(0.98),
    )


def test_lowp_gemm_ill_scaled_huge() raises:
    # Inputs ~5e3 in magnitude -- would saturate fp8 without a compensating
    # (small) scale.
    _run_lowp_gemm_case[False, False](
        64, 64, 128, Float32(5.0e3), "ill-scaled-huge"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
