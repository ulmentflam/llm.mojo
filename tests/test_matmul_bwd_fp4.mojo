# ===----------------------------------------------------------------------=== #
# tests/test_matmul_bwd_fp4.mojo — Chunk T2a gate (c) + accumulate unit test:
#
#   1. `matmul_bwd_fp4` (llmm/matmul.mojo, Chunk T2a: Dgrad = SR on d_output,
#      RNE on weight, no RHT; Wgrad = RNE/SR, no RHT yet — RHT lands in T2b)
#      vs the production bf16 `matmul_bwd` at the two FP4-eligible MLP linear
#      shapes (fc, proj), real GPT-2 124M (d12, channels=768) dimensions and
#      the real B=4/T=1024 row count (rows=4096) — mirrors
#      tests/test_matmul_fwd_fp4.mojo's gate (c) methodology for the backward
#      pass. Compares `d_input`/`d_weight` (relL2 + cosine, NaN/Inf-free) and
#      asserts `d_bias` is BIT-IDENTICAL between the two arms (both paths
#      reuse the same bf16 `matmul_bias_bwd` kernel — the "bias-grad reuse"
#      design decision — so any divergence there would flag a wiring bug,
#      not ordinary fp4 quantization noise).
#   2. `test_fp4_gemm_accumulate*` — unit tests for the `lowp_gemm_fp4`
#      accumulate fix (Chunk T2a, mechanics item 2): a fresh (accumulate=
#      False) GEMM result added to a known seed value via a SEPARATE
#      accumulate=True call must match seeding the accumulator directly and
#      adding, to within bf16 rounding — mirrors the fp8 beta=1 probe cited
#      in docs/ai/low_precision_gotchas.md C4.
#
# GPU-only tests are guarded by `has_nvidia_gpu_accelerator()` and expected to
# run under `flock -w 10800 /tmp/llmm-gpu.lock -c '...'` (shared GPU).
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import sqrt
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true, assert_equal

from llmm.matmul import (
    matmul_bwd,
    matmul_bwd_fp4,
    lowp_gemm_fp4,
)
from llmm.nvfp4_quant import nvfp4_packed_size, nvfp4_scale_buffer_size
from llmm.memory import MutKernelPtr, ImmutKernelPtr


# ===----------------------------------------------------------------------=== #
# 1. matmul_bwd_fp4 vs matmul_bwd (bf16) at MLP shapes.
# ===----------------------------------------------------------------------=== #


def _fill_random(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    numel: Int,
    scale: Float32,
) -> None:
    for i in range(numel):
        var v = Float32((random_float64() * 2.0 - 1.0)) * scale
        ptr[i] = v.cast[DType.bfloat16]()


def _rel_l2_cosine(
    got: UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    want: UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    n: Int,
    label: String,
    mut rel_l2_out: Float32,
    mut cosine_out: Float32,
) raises -> None:
    var l2_err = Float32(0.0)
    var dot = Float32(0.0)
    var norm_got = Float32(0.0)
    var norm_want = Float32(0.0)
    for i in range(n):
        var g = got[i].cast[DType.float32]()
        var w = want[i].cast[DType.float32]()
        assert_true(g == g, label + ": NaN at " + String(i))
        assert_true(
            g > Float32(-1e30) and g < Float32(1e30),
            label + ": Inf/overflow at " + String(i),
        )
        var e = g - w
        l2_err += e * e
        dot += g * w
        norm_got += g * g
        norm_want += w * w
    rel_l2_out = sqrt(l2_err / (norm_want + Float32(1e-12)))
    cosine_out = dot / (sqrt(norm_got) * sqrt(norm_want) + Float32(1e-12))


def _run_bwd_site_case(
    label: String,
    rows: Int,
    in_channels: Int,
    out_channels: Int,
    use_gelu: Bool,
) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime DT = DType.bfloat16

    var n_in = rows * in_channels
    var n_w = out_channels * in_channels
    var n_out = rows * out_channels

    seed(20260710)
    var host_input = ctx.enqueue_create_host_buffer[DT](n_in)
    var host_weight = ctx.enqueue_create_host_buffer[DT](n_w)
    var host_bias = ctx.enqueue_create_host_buffer[DT](out_channels)
    var host_doutput = ctx.enqueue_create_host_buffer[DT](n_out)
    var host_pre_gelu = ctx.enqueue_create_host_buffer[DT](n_in)
    _fill_random(host_input.unsafe_ptr(), n_in, Float32(1.5))
    _fill_random(host_weight.unsafe_ptr(), n_w, Float32(0.02))
    _fill_random(host_bias.unsafe_ptr(), out_channels, Float32(0.01))
    _fill_random(host_doutput.unsafe_ptr(), n_out, Float32(0.05))
    _fill_random(host_pre_gelu.unsafe_ptr(), n_in, Float32(1.0))

    var dev_input = ctx.enqueue_create_buffer[DT](n_in)
    var dev_weight = ctx.enqueue_create_buffer[DT](n_w)
    var dev_bias = ctx.enqueue_create_buffer[DT](out_channels)
    var dev_doutput = ctx.enqueue_create_buffer[DT](n_out)
    var dev_pre_gelu = ctx.enqueue_create_buffer[DT](n_in)
    dev_input.enqueue_copy_from(host_input)
    dev_weight.enqueue_copy_from(host_weight)
    dev_bias.enqueue_copy_from(host_bias)
    dev_doutput.enqueue_copy_from(host_doutput)
    dev_pre_gelu.enqueue_copy_from(host_pre_gelu)
    ctx.synchronize()

    # bf16 reference (matmul_bwd, accumulate=False so both arms overwrite
    # cleanly -- accumulate mechanics are covered separately below).
    var d_input_ref = ctx.enqueue_create_buffer[DT](n_in)
    var d_weight_ref = ctx.enqueue_create_buffer[DT](n_w)
    var d_bias_ref = ctx.enqueue_create_buffer[DT](out_channels)
    var scratch_ref = ctx.enqueue_create_buffer[DT](n_out)
    ctx.synchronize()

    if use_gelu:
        matmul_bwd[DT, "gpu", use_gelu=True, accumulate=False, has_bias=True](
            d_input_ref.unsafe_ptr().as_unsafe_any_origin(),
            d_weight_ref.unsafe_ptr().as_unsafe_any_origin(),
            d_bias_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_doutput.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_pre_gelu.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            scratch_ref.unsafe_ptr().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    else:
        matmul_bwd[DT, "gpu", use_gelu=False, accumulate=False, has_bias=True](
            d_input_ref.unsafe_ptr().as_unsafe_any_origin(),
            d_weight_ref.unsafe_ptr().as_unsafe_any_origin(),
            d_bias_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_doutput.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_pre_gelu.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            scratch_ref.unsafe_ptr().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    ctx.synchronize()

    # fp4 arm (matmul_bwd_fp4, Chunk T2a).
    var d_input_fp4 = ctx.enqueue_create_buffer[DT](n_in)
    var d_weight_fp4 = ctx.enqueue_create_buffer[DT](n_w)
    var d_bias_fp4 = ctx.enqueue_create_buffer[DT](out_channels)
    ctx.synchronize()

    if use_gelu:
        matmul_bwd_fp4[
            DT, "gpu", use_gelu=True, accumulate=False, has_bias=True
        ](
            d_input_fp4.unsafe_ptr().as_unsafe_any_origin(),
            d_weight_fp4.unsafe_ptr().as_unsafe_any_origin(),
            d_bias_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_doutput.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_pre_gelu.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    else:
        matmul_bwd_fp4[
            DT, "gpu", use_gelu=False, accumulate=False, has_bias=True
        ](
            d_input_fp4.unsafe_ptr().as_unsafe_any_origin(),
            d_weight_fp4.unsafe_ptr().as_unsafe_any_origin(),
            d_bias_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_doutput.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_pre_gelu.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    ctx.synchronize()

    var host_d_input_ref = ctx.enqueue_create_host_buffer[DT](n_in)
    var host_d_input_fp4 = ctx.enqueue_create_host_buffer[DT](n_in)
    var host_d_weight_ref = ctx.enqueue_create_host_buffer[DT](n_w)
    var host_d_weight_fp4 = ctx.enqueue_create_host_buffer[DT](n_w)
    var host_d_bias_ref = ctx.enqueue_create_host_buffer[DT](out_channels)
    var host_d_bias_fp4 = ctx.enqueue_create_host_buffer[DT](out_channels)
    d_input_ref.enqueue_copy_to(host_d_input_ref)
    d_input_fp4.enqueue_copy_to(host_d_input_fp4)
    d_weight_ref.enqueue_copy_to(host_d_weight_ref)
    d_weight_fp4.enqueue_copy_to(host_d_weight_fp4)
    d_bias_ref.enqueue_copy_to(host_d_bias_ref)
    d_bias_fp4.enqueue_copy_to(host_d_bias_fp4)
    ctx.synchronize()

    # d_bias: bit-identical (same bf16 matmul_bias_bwd kernel on both arms).
    for i in range(out_channels):
        assert_equal(
            host_d_bias_ref.unsafe_ptr()[i],
            host_d_bias_fp4.unsafe_ptr()[i],
            label
            + ": d_bias diverged at "
            + String(i)
            + " (bias-grad reuse should make fp4/bf16 d_bias bit-identical)",
        )

    var rel_l2_input = Float32(0.0)
    var cosine_input = Float32(0.0)
    _rel_l2_cosine(
        host_d_input_fp4.unsafe_ptr(),
        host_d_input_ref.unsafe_ptr(),
        n_in,
        label + " d_input",
        rel_l2_input,
        cosine_input,
    )
    var rel_l2_weight = Float32(0.0)
    var cosine_weight = Float32(0.0)
    _rel_l2_cosine(
        host_d_weight_fp4.unsafe_ptr(),
        host_d_weight_ref.unsafe_ptr(),
        n_w,
        label + " d_weight",
        rel_l2_weight,
        cosine_weight,
    )
    print(
        label
        + ": d_input rel_l2="
        + String(rel_l2_input)
        + " cosine="
        + String(cosine_input)
        + " | d_weight rel_l2="
        + String(rel_l2_weight)
        + " cosine="
        + String(cosine_weight)
    )

    # Bounds per the F1 gotcha (docs/ai/low_precision_gotchas.md): the
    # cosine floor is DERIVED from the calibrated relL2 floor via
    # cosine ~= 1/sqrt(1+relL2^2), not copy-pasted from a higher-precision
    # sibling. Measured (three runs, bit-identical): fc/proj d_input relL2
    # 0.1836/0.1839, d_weight relL2 0.1784/0.1782, cosine 0.9845-0.9850 --
    # a bit worse than the forward pass's single-GEMM ~0.151 floor (commit
    # cac1968), as expected: Dgrad/Wgrad each compose TWO fp4 quantizations
    # (weight/input RNE + d_output SR) through one GEMM, plus SR's own
    # dither variance on top of RNE's rounding error. relL2 < 0.22 leaves
    # ~20% headroom above the measured floor; the cosine floor derives from
    # that bound (1/sqrt(1+0.22^2) ~= 0.977), with a small margin (0.975)
    # matching cac1968's precedent of sitting just below the geometric
    # value rather than exactly on it.
    assert_true(
        rel_l2_input < Float32(0.22),
        label + ": d_input rel_l2 " + String(rel_l2_input) + " >= 0.22",
    )
    assert_true(
        cosine_input > Float32(0.975),
        label + ": d_input cosine " + String(cosine_input) + " <= 0.975",
    )
    assert_true(
        rel_l2_weight < Float32(0.22),
        label + ": d_weight rel_l2 " + String(rel_l2_weight) + " >= 0.22",
    )
    assert_true(
        cosine_weight > Float32(0.975),
        label + ": d_weight cosine " + String(cosine_weight) + " <= 0.975",
    )


def test_fc_bwd_site() raises:
    # fc backward (use_gelu=False, DGELU already folded into proj's
    # backward): C=768, OC=3072, rows=4096 (B4*T1024).
    _run_bwd_site_case("fc-bwd", 4096, 768, 4 * 768, use_gelu=False)


def test_proj_bwd_site() raises:
    # proj backward (use_gelu=True, fuses DGELU at the 4C boundary):
    # C=3072, OC=768.
    _run_bwd_site_case("proj-bwd", 4096, 4 * 768, 768, use_gelu=True)


# ===----------------------------------------------------------------------=== #
# 2. lowp_gemm_fp4 accumulate=True unit test (Chunk T2a, mechanics item 2).
# ===----------------------------------------------------------------------=== #


def test_fp4_gemm_accumulate() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime DT = DType.bfloat16

    var m = 64
    var n = 64
    var k = 64

    seed(20260710)
    var host_a = ctx.enqueue_create_host_buffer[DT](m * k)
    var host_b = ctx.enqueue_create_host_buffer[DT](n * k)
    _fill_random(host_a.unsafe_ptr(), m * k, Float32(1.0))
    _fill_random(host_b.unsafe_ptr(), n * k, Float32(1.0))

    var dev_a = ctx.enqueue_create_buffer[DT](m * k)
    var dev_b = ctx.enqueue_create_buffer[DT](n * k)
    dev_a.enqueue_copy_from(host_a)
    dev_b.enqueue_copy_from(host_b)
    ctx.synchronize()

    var a_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(m, k))
    var a_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(m, k, 1)
    )
    var a_tscale = ctx.enqueue_create_buffer[DType.float32](1)
    var b_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(n, k))
    var b_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(n, k, 1)
    )
    var b_tscale = ctx.enqueue_create_buffer[DType.float32](1)

    # Fresh (accumulate=False) result -- the "expected fresh contribution".
    var dev_d_fresh = ctx.enqueue_create_buffer[DT](m * n)
    ctx.synchronize()
    lowp_gemm_fp4[DT, DT, "gpu", 1, 1](
        rebind[MutKernelPtr[DT]](
            dev_d_fresh.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DT]](
            dev_d_fresh.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DT]](
            dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DT]](
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            a_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            b_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        m,
        n,
        k,
        ctx,
    )
    ctx.synchronize()

    # Accumulate=True into a buffer pre-seeded with a known constant, using a
    # SEPARATE raw-scratch buffer (required whenever accumulate=True -- see
    # lowp_gemm_fp4's docstring).
    comptime SEED_VAL = Float32(2.0)
    var host_seed = ctx.enqueue_create_host_buffer[DT](m * n)
    for i in range(m * n):
        host_seed.unsafe_ptr()[i] = SEED_VAL.cast[DT]()
    var dev_d_acc = ctx.enqueue_create_buffer[DT](m * n)
    dev_d_acc.enqueue_copy_from(host_seed)
    var dev_d_raw_scratch = ctx.enqueue_create_buffer[DT](m * n)
    ctx.synchronize()

    lowp_gemm_fp4[DT, DT, "gpu", 1, 1, accumulate=True](
        rebind[MutKernelPtr[DT]](dev_d_acc.unsafe_ptr().as_unsafe_any_origin()),
        rebind[MutKernelPtr[DT]](
            dev_d_raw_scratch.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DT]](
            dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DT]](
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            a_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            b_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        m,
        n,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_fresh = ctx.enqueue_create_host_buffer[DT](m * n)
    var host_acc = ctx.enqueue_create_host_buffer[DT](m * n)
    dev_d_fresh.enqueue_copy_to(host_fresh)
    dev_d_acc.enqueue_copy_to(host_acc)
    ctx.synchronize()

    var max_abs_diff = Float32(0.0)
    for i in range(m * n):
        var fresh = host_fresh.unsafe_ptr()[i].cast[DType.float32]()
        var acc = host_acc.unsafe_ptr()[i].cast[DType.float32]()
        assert_true(acc == acc, "accumulate: NaN at " + String(i))
        var want = SEED_VAL + fresh
        var diff = acc - want
        if diff < 0.0:
            diff = -diff
        if diff > max_abs_diff:
            max_abs_diff = diff
    print("fp4-gemm-accumulate: max_abs_diff vs (seed + fresh) =", max_abs_diff)
    # bf16 rounding of the seed (2.0, exact) + fresh (up to a few units) is
    # small; a broken accumulate (e.g. double-counting, or reading the
    # post-overwrite raw buffer instead of the pre-existing d_ptr value)
    # would produce an O(1)-or-larger systematic error, not this tight a
    # band -- see docs/ai/low_precision_gotchas.md C4's fp8 beta=1 probe for
    # the analogous "to within bf16 rounding" bound.
    assert_true(
        max_abs_diff < Float32(0.05),
        "fp4-gemm-accumulate: max abs diff "
        + String(max_abs_diff)
        + " >= 0.05 -- accumulate is not adding correctly",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
