# ===----------------------------------------------------------------------=== #
# test_amax.mojo — Chunk C gate (docs/ai/fp8_training_design.md, "Gate C").
#
# Covers: (1) scale formula (known amax history -> expected scale, e4m3/e5m2,
# margin, clamping); (2) warmup uses current scaling, not history; (3) scales
# stay finite/non-zero across a spike and across the all-zero/NaN/Inf edge
# cases; (4) ring buffer wraps correctly; (5) determinism — identical inputs
# give bit-identical amax/scale on repeat, both for `compute_amax`'s GPU
# reduction and `AmaxState.update_scale`'s state machine.
#
# GPU-only (see llmm/amax.mojo's module docstring) — every test guards with
# `has_nvidia_gpu_accelerator()` and returns early (skips) if absent, matching
# tests/test_zero.mojo's convention. Run under
# `flock -w 10800 /tmp/llmm-gpu.lock -c 'pixi run mojo run -I . tests/test_amax.mojo'`
# per the shared-GPU convention (docs/ai/... / AGENTS.md).
# ===----------------------------------------------------------------------=== #

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.sys import has_nvidia_gpu_accelerator
from std.math import isnan, isinf
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from llmm.amax import (
    format_max,
    compute_amax,
    AmaxState,
    kernel_ptr_as_immut,
    device_buf_mut_ptr,
)
from llmm.lowp import PrecisionSpec, ScalingKind, FP8_SPEC

from _lowp_test_common import make_bf16_tensor as _make_bf16_tensor


# ===----------------------------------------------------------------------=== #
# Small host<->device scalar/tensor helpers (mirrors tests/test_zero.mojo).
# ===----------------------------------------------------------------------=== #


def _read_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32]
) raises -> Float32:
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    buf.enqueue_copy_to(host)
    ctx.synchronize()
    return host.unsafe_ptr()[0]


def _write_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], val: Float32
) raises -> None:
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    host.unsafe_ptr()[0] = val
    buf.enqueue_copy_from(host)
    ctx.synchronize()


def _make_fp32_tensor(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr()[i] = values[i]
    var dev = ctx.enqueue_create_buffer[DType.float32](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def _compute_amax_bf16(
    ctx: DeviceContext, values: List[Float32]
) raises -> Float32:
    var data = _make_bf16_tensor(ctx, values)
    var out = ctx.enqueue_create_buffer[DType.float32](1)
    compute_amax[FP8_SPEC, DType.bfloat16](
        device_buf_mut_ptr(out),
        kernel_ptr_as_immut(device_buf_mut_ptr(data)),
        len(values),
        ctx,
    )
    ctx.synchronize()
    return _read_f32(ctx, out)


# A small custom spec (independent of FP8_SPEC) to exercise a nonzero margin
# and a short history length without perturbing the FP8_SPEC constant other
# tests rely on. Field order matches llmm/lowp.mojo's `PrecisionSpec`
# (fwd_dtype, bwd_dtype, scale_dtype, scaling, block, amax_history_len,
# margin, stochastic_rounding, hadamard).
comptime _MARGIN1_SPEC = PrecisionSpec(
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float32,
    ScalingKind.PerTensor,
    0,
    4,
    1,
    False,
    False,
)

comptime _H4_SPEC = PrecisionSpec(
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float32,
    ScalingKind.PerTensor,
    0,
    4,
    0,
    False,
    False,
)


# ===----------------------------------------------------------------------=== #
# format_max
# ===----------------------------------------------------------------------=== #


def test_format_max_e4m3() raises:
    assert_equal(format_max[DType.float8_e4m3fn](), Float32(448.0))


def test_format_max_e5m2() raises:
    assert_equal(format_max[DType.float8_e5m2](), Float32(57344.0))


def test_format_max_e2m1_seam() raises:
    # FP4's e2m1 max (6.0) slots in as a constant even though FP4 quantize
    # itself is an unbuilt seam (docs/ai/fp8_training_design.md §3).
    assert_equal(format_max[DType.float4_e2m1fn](), Float32(6.0))


# ===----------------------------------------------------------------------=== #
# compute_amax — correctness, all-zero, NaN/Inf detection, determinism.
# ===----------------------------------------------------------------------=== #


def test_compute_amax_bf16_basic() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.append(1.0)
    values.append(-7.5)
    values.append(3.25)
    values.append(-2.0)
    var amax = _compute_amax_bf16(ctx, values)
    # bf16 round-trip of 7.5 is exact (7.5 = 1.111b * 2^2, well within bf16's
    # 8-bit mantissa), so this should match the fp32 reference exactly.
    assert_almost_equal(amax, Float32(7.5), atol=1e-3)


def test_compute_amax_fp32_basic() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.append(-100.0)
    values.append(4.0)
    values.append(99.5)
    var data = _make_fp32_tensor(ctx, values)
    var out = ctx.enqueue_create_buffer[DType.float32](1)
    compute_amax[FP8_SPEC, DType.float32](
        device_buf_mut_ptr(out),
        kernel_ptr_as_immut(device_buf_mut_ptr(data)),
        len(values),
        ctx,
    )
    ctx.synchronize()
    var amax = _read_f32(ctx, out)
    assert_almost_equal(amax, Float32(100.0), atol=1e-4)


def test_compute_amax_large_tensor() raises:
    # Exercise the multi-block grid-stride reduction path (not just a
    # single-block toy size) so the two-kernel partial/aggregate structure is
    # actually tested, not just its degenerate 1-block case.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime n = 200_000
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host.unsafe_ptr()[i] = Float32(0.0).cast[DType.bfloat16]()
    # Plant the true max at an arbitrary, non-block-aligned interior index.
    host.unsafe_ptr()[123457] = Float32(-42.0).cast[DType.bfloat16]()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()

    var out = ctx.enqueue_create_buffer[DType.float32](1)
    compute_amax[FP8_SPEC, DType.bfloat16](
        device_buf_mut_ptr(out),
        kernel_ptr_as_immut(device_buf_mut_ptr(dev)),
        n,
        ctx,
    )
    ctx.synchronize()
    var amax = _read_f32(ctx, out)
    assert_almost_equal(amax, Float32(42.0), atol=1e-3)


def test_compute_amax_all_zero() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.0)
    values.append(0.0)
    var amax = _compute_amax_bf16(ctx, values)
    assert_equal(amax, Float32(0.0))
    assert_true(not isnan(amax))
    assert_true(not isinf(amax))


def test_compute_amax_nan_detection() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.append(1.0)
    values.append(Float32(0.0) / Float32(0.0))  # NaN
    values.append(2.0)
    var amax = _compute_amax_bf16(ctx, values)
    assert_true(isnan(amax))


def test_compute_amax_inf_detection() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.append(1.0)
    values.append(Float32(1.0) / Float32(0.0))  # +Inf
    values.append(2.0)
    var amax = _compute_amax_bf16(ctx, values)
    assert_true(isnan(amax))


def test_compute_amax_determinism() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var values = List[Float32]()
    for i in range(4096):
        values.append(Float32((i * 2654435761) % 10007) * 0.01 - 25.0)
    var a = _compute_amax_bf16(ctx, values)
    var b = _compute_amax_bf16(ctx, values)
    assert_equal(a, b)


# ===----------------------------------------------------------------------=== #
# AmaxState / update_scale — formula, warmup, clamping, ring buffer, determinism.
# ===----------------------------------------------------------------------=== #


def test_update_scale_warmup_uses_current_amax() raises:
    # Per docs/ai/fp8_training_design.md §1.3: during warmup (the first
    # `amax_history_len` calls), scale is derived from *this* call's amax,
    # not history — even when history would suggest something different.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)

    # H = 4, margin = 0: scale = fmt_max / amax_current, every one of the
    # first 4 calls (each strictly "current scaling", ignoring history).
    var inputs = List[Float32]()
    inputs.append(10.0)
    inputs.append(1.0)
    inputs.append(100.0)
    inputs.append(4.0)
    comptime fmt_max = Float32(448.0)  # e4m3fn

    for i in range(4):
        _write_f32(ctx, amax_current, inputs[i])
        state.update_scale[DType.float8_e4m3fn](
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
        )
        ctx.synchronize()
        var scale = _read_f32(ctx, state.scale)
        var expected = fmt_max / inputs[i]
        assert_almost_equal(scale, expected, rtol=1e-5)


def test_update_scale_steady_state_uses_history_max() raises:
    # After warmup (H=4), the scale must come from max(history), *not* the
    # just-pushed current-step amax — this is what makes it "delayed".
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    comptime fmt_max = Float32(448.0)

    # Warm up with a big spike at step 0, then small values at 1..3.
    var warmup = List[Float32]()
    warmup.append(80.0)
    warmup.append(2.0)
    warmup.append(3.0)
    warmup.append(4.0)
    for i in range(4):
        _write_f32(ctx, amax_current, warmup[i])
        state.update_scale[DType.float8_e4m3fn](
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
        )
        ctx.synchronize()

    # Steady-state call: current amax is tiny (1.0), but history still
    # contains the 80.0 spike from step 0 -> scale must reflect max(history)
    # = 80.0, not the tiny current value.
    _write_f32(ctx, amax_current, 1.0)
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    var expected = fmt_max / Float32(80.0)
    assert_almost_equal(scale, expected, rtol=1e-5)


def test_update_scale_margin() raises:
    # scale = fmt_max / (amax / 2^margin) = fmt_max * 2^margin / amax.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_MARGIN1_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    _write_f32(ctx, amax_current, 10.0)
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    var expected = Float32(448.0) * Float32(2.0) / Float32(10.0)
    assert_almost_equal(scale, expected, rtol=1e-5)


def test_update_scale_e5m2_format_max() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    _write_f32(ctx, amax_current, 1000.0)
    state.update_scale[DType.float8_e5m2](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    var expected = Float32(57344.0) / Float32(1000.0)
    assert_almost_equal(scale, expected, rtol=1e-5)


def test_update_scale_all_zero_amax_no_div_by_zero() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    _write_f32(ctx, amax_current, 0.0)
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    var scale_inv = _read_f32(ctx, state.scale_inv)
    assert_true(not isnan(scale) and not isinf(scale))
    assert_true(not isnan(scale_inv) and not isinf(scale_inv))
    assert_almost_equal(scale, Float32(1.0), atol=1e-6)


def test_update_scale_nan_amax_no_nan_scale() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    _write_f32(ctx, amax_current, Float32(0.0) / Float32(0.0))
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    assert_true(not isnan(scale) and not isinf(scale))
    assert_almost_equal(scale, Float32(1.0), atol=1e-6)


def test_update_scale_inf_amax_no_inf_scale() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    _write_f32(ctx, amax_current, Float32(1.0) / Float32(0.0))
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    assert_true(not isnan(scale) and not isinf(scale))
    assert_almost_equal(scale, Float32(1.0), atol=1e-6)


def test_update_scale_ring_buffer_wraps() raises:
    # H = 4. After 8 calls (2 full wraps), the history must hold only the
    # last 4 pushed values -- an early spike must have rotated out.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    comptime fmt_max = Float32(448.0)

    # Steps 0..3 (warmup): spike at step 0, small afterward.
    var seq = List[Float32]()
    seq.append(90.0)  # step 0 -- should be rotated out by step 8
    seq.append(2.0)  # step 1
    seq.append(2.0)  # step 2
    seq.append(2.0)  # step 3
    # Steps 4..7 (steady state): small values, no spike.
    seq.append(2.0)  # step 4
    seq.append(2.0)  # step 5
    seq.append(2.0)  # step 6
    seq.append(3.0)  # step 7 -- history after this call = {2,2,2,3}

    for i in range(8):
        _write_f32(ctx, amax_current, seq[i])
        state.update_scale[DType.float8_e4m3fn](
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
        )
        ctx.synchronize()

    # One more steady-state call: current amax is irrelevant to *this* call's
    # scale (delayed scaling), and the 90.0 spike from step 0 must no longer
    # be present in history (H=4, and 4 steady-state pushes have happened
    # since: steps 4,5,6,7 all overwrote index 0's slot at least once).
    _write_f32(ctx, amax_current, 1.0)
    state.update_scale[DType.float8_e4m3fn](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
    )
    ctx.synchronize()
    var scale = _read_f32(ctx, state.scale)
    # max(history going into this call) = max({2,2,2,3}) = 3.0, not 90.0.
    var expected = fmt_max / Float32(3.0)
    assert_almost_equal(scale, expected, rtol=1e-5)


def _run_update_scale_sequence(
    ctx: DeviceContext, seq: List[Float32]
) raises -> Float32:
    var state = AmaxState[_H4_SPEC](ctx)
    var amax_current = ctx.enqueue_create_buffer[DType.float32](1)
    for i in range(len(seq)):
        _write_f32(ctx, amax_current, seq[i])
        state.update_scale[DType.float8_e4m3fn](
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_current)), ctx
        )
        ctx.synchronize()
    return _read_f32(ctx, state.scale)


def test_update_scale_determinism() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()

    var seq = List[Float32]()
    for i in range(40):
        seq.append(Float32((i * 37 + 5) % 97) + 1.0)

    var a = _run_update_scale_sequence(ctx, seq)
    var b = _run_update_scale_sequence(ctx, seq)
    assert_equal(a, b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
