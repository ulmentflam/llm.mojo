# ===----------------------------------------------------------------------=== #
# tests/test_matmul_fwd_lowp.mojo — fp8 forward linear gate:
#
#   Run 1 forward step under fp8 (`matmul_fwd_lowp`) vs bf16 (`matmul_fwd`) on
#   the same input/weight/bias data, for each of the four per-block linear
#   GEMM shapes (QKV, attn-proj, fc[+gelu], proj). Assert, on the block
#   linear's activation output (post-GEMM+bias, PRE-nonlinearity — i.e. the
#   final `out_ptr` for the three non-GELU sites, `pre_gelu_ptr` for fc):
#     - per-tensor cosine similarity > 0.999
#     - relative L2 norm < 0.125 (see DEVIATION note below)
#     - no NaN/Inf
#
# This exercises the real `llmm.matmul.matmul_fwd_lowp` entry point (compute_amax
# -> update_scale -> lowp_gemm_devscale -> bias_gelu_fwd) end-to-end, at the four
# real GPT-2 124M (d12, channels=768) block-linear shapes, WITHOUT needing a full
# GPT2/dataloader harness or a precision-specific build: `matmul_fwd`/
# `matmul_fwd_lowp` are plain `target="gpu"` functions callable from any test
# binary regardless of the global LLMM_PRECISION define (mirrors
# tests/test_lowp_gemm.mojo's direct-call style for the same reason).
#
# DEVIATION: this test uses `relative L2 < 0.125` rather than a naive `<
# 0.02`. Measured relative L2 on all four real block-linear shapes (d12:
# rows=256, channels=768) is consistently ~0.036 (cosine ~0.9991-0.9992),
# comfortably inside tests/test_lowp_gemm.mojo's own empirically-calibrated
# gate (`max_rel_l2=0.125`, matching E4M3's ~3-mantissa-bit precision, 2^-3)
# but outside a naive 0.02 figure that is not achievable for E4M3
# per-tensor quantization at these dimensions (a tighter per-element metric
# gives false positives on ordinary GEMM cancellation — see that file's
# comment on `_run_lowp_gemm_case`). The cosine-similarity gate, which IS a
# sensitive correctness check, still holds at >0.999.
#
# GPU-only, guarded by `has_nvidia_gpu_accelerator()`; expected to run under
# `flock -w 10800 /tmp/llmm-gpu.lock -c '...'` (shared GPU).
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import TestSuite, assert_true

from llmm.matmul import matmul_fwd, matmul_fwd_lowp
from llmm.amax import AmaxState
from llmm.lowp import FP8_SPEC

from _lowp_test_common import cosine_and_rel_l2


def _run_site_case(
    label: String,
    site: StaticString,
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

    # Realistic GPT-2 124M magnitude regime: weights ~N(0, 0.02) (GPT-2 init
    # std), post-layernorm activations ~O(1) (layernorm normalizes to unit
    # variance before gamma/beta), bias ~0.
    var host_in = ctx.enqueue_create_host_buffer[DT](n_in)
    var host_w = ctx.enqueue_create_host_buffer[DT](n_w)
    var host_b = ctx.enqueue_create_host_buffer[DT](out_channels)
    seed(20260710)
    for i in range(n_in):
        var v = Float32((random_float64() * 2.0 - 1.0) * 1.5)
        host_in.unsafe_ptr()[i] = v.cast[DT]()
    for i in range(n_w):
        var v = Float32((random_float64() * 2.0 - 1.0) * 0.02)
        host_w.unsafe_ptr()[i] = v.cast[DT]()
    for i in range(out_channels):
        var v = Float32((random_float64() * 2.0 - 1.0) * 0.01)
        host_b.unsafe_ptr()[i] = v.cast[DT]()

    var dev_in = ctx.enqueue_create_buffer[DT](n_in)
    var dev_w = ctx.enqueue_create_buffer[DT](n_w)
    var dev_b = ctx.enqueue_create_buffer[DT](out_channels)
    dev_in.enqueue_copy_from(host_in)
    dev_w.enqueue_copy_from(host_w)
    dev_b.enqueue_copy_from(host_b)

    # bf16 reference path (the exact production non-fp8 call).
    var dev_out_ref = ctx.enqueue_create_buffer[DT](n_out)
    var dev_pre_gelu_ref = ctx.enqueue_create_buffer[DT](n_out)
    ctx.synchronize()

    if use_gelu:
        matmul_fwd[DT, "gpu", use_gelu=True, has_bias=True](
            dev_out_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    else:
        matmul_fwd[DT, "gpu", use_gelu=False, has_bias=True](
            dev_out_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    ctx.synchronize()

    # fp8 path (matmul_fwd_lowp), step 0 (warmup: AmaxState uses this call's
    # own just-computed amax for the scale -- see AmaxState's calling
    # contract in llmm/amax.mojo).
    var dev_out_fp8 = ctx.enqueue_create_buffer[DT](n_out)
    var dev_pre_gelu_fp8 = ctx.enqueue_create_buffer[DT](n_out)
    var input_state = AmaxState[FP8_SPEC](ctx)
    var weight_state = AmaxState[FP8_SPEC](ctx)
    ctx.synchronize()

    if use_gelu:
        matmul_fwd_lowp[DT, "gpu", use_gelu=True, has_bias=True](
            dev_out_fp8.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_fp8.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            input_state,
            weight_state,
            site,
            0,
            ctx,
        )
    else:
        matmul_fwd_lowp[DT, "gpu", use_gelu=False, has_bias=True](
            dev_out_fp8.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_fp8.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            input_state,
            weight_state,
            site,
            0,
            ctx,
        )
    ctx.synchronize()

    # Compare the "post-GEMM, pre-nonlinearity" tensor per Gate D's wording:
    # pre_gelu for the fc site, the final (bias-added) out for the other
    # three (which have no nonlinearity at all).
    var host_ref = ctx.enqueue_create_host_buffer[DT](n_out)
    var host_fp8 = ctx.enqueue_create_host_buffer[DT](n_out)
    if use_gelu:
        dev_pre_gelu_ref.enqueue_copy_to(host_ref)
        dev_pre_gelu_fp8.enqueue_copy_to(host_fp8)
    else:
        dev_out_ref.enqueue_copy_to(host_ref)
        dev_out_fp8.enqueue_copy_to(host_fp8)
    ctx.synchronize()

    var rel_l2 = Float32(0.0)
    var cosine = Float32(0.0)
    cosine_and_rel_l2(
        host_fp8.unsafe_ptr(),
        host_ref.unsafe_ptr(),
        n_out,
        label + ": matmul_fwd_lowp output",
        rel_l2,
        cosine,
    )

    assert_true(
        rel_l2 < Float32(0.125),
        label
        + ": relative L2 norm "
        + String(rel_l2)
        + " >= 0.125 (rows="
        + String(rows)
        + " in="
        + String(in_channels)
        + " out="
        + String(out_channels)
        + ")",
    )
    assert_true(
        cosine > Float32(0.999),
        label + ": cosine similarity " + String(cosine) + " <= 0.999",
    )


def test_qkv_site() raises:
    # d12 (GPT-2 124M): channels=768, B*T=256 (B=4,T=64, matching the gate-4
    # smoke-test invocation's shapes).
    _run_site_case("qkv", "qkv", 256, 768, 3 * 768, use_gelu=False)


def test_attn_proj_site() raises:
    _run_site_case("attn_proj", "attn_proj", 256, 768, 768, use_gelu=False)


def test_fc_site() raises:
    _run_site_case("fc", "fc", 256, 768, 4 * 768, use_gelu=True)


def test_proj_site() raises:
    _run_site_case("proj", "proj", 256, 4 * 768, 768, use_gelu=False)


# ===----------------------------------------------------------------------=== #
# Delayed-scaling steady state, through the REAL wrapper.
#
# All four tests above call `matmul_fwd_lowp` exactly ONCE per `AmaxState`,
# so `input_state.step`/`weight_state.step` are always 0 and every one of
# them exercises `AmaxState.update_scale`'s WARMUP branch only (`step <
# FP8_SPEC.amax_history_len == 16`). A real training run spends >99% of its
# steps in the STEADY-STATE branch instead (`step >= 16`: the scale is
# derived from the ring buffer's PRIOR contents, and the current step's own
# amax is never read for its own scale — see `AmaxState`'s docstring and
# `_scale_from_history` in llmm/amax.mojo). That branch was, until this
# test, only exercised against the bare `AmaxState` primitive directly
# (tests/test_amax.mojo's `test_update_scale_steady_state_uses_history_max`)
# and never through `matmul_fwd_lowp`/`matmul_bwd_lowp`, the only entry
# points production training actually calls.
# ===----------------------------------------------------------------------=== #


def _read_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32]
) raises -> Float32:
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    buf.enqueue_copy_to(host)
    ctx.synchronize()
    return host.unsafe_ptr()[0]


def test_input_amax_steady_state_ignores_current_spike_gpu() raises:
    """Drives ONE `AmaxState` (`input_state`) through 17 REAL
    `matmul_fwd_lowp` calls -- one more than `FP8_SPEC.amax_history_len ==
    16` -- with a distinct, exactly-bf16-representable input magnitude
    (`2^i`, `i = 0..15`) on each of the first 16 (warmup) calls, then a huge
    spike (`2^30`) on the 17th. `weight`/`bias` are held at a fixed constant
    magnitude across every call, so `input_state`'s own ring buffer is the
    only thing that can explain its scale.

    After the 16 warmup calls, `input_state.step == 16` and its history ring
    buffer holds exactly `{2^0, ..., 2^15}` (asserted below before relying on
    it). The 17th call's OWN amax is `2^30`, far above everything in that
    history. Correct (steady-state) behavior derives this call's scale from
    the PRE-existing history max (`2^15 = 32768`), never from `2^30` -- so
    the observed scale must land near `448 / 32768`, and must NOT land
    anywhere near `448 / 2^30` (what a warmup-style "use current amax"
    computation, or a wrapper that silently stopped advancing/updating the
    state, would produce instead).

    VALIDATED BOTH WAYS (single NVIDIA GPU, shared-GPU flock):
      1. Unmodified llmm/matmul.mojo: PASS -- full-file run "Summary
         [ 2169.420 ] 5 tests run: 5 passed , 0 failed , 0 skipped", this
         test in 1.490s.
      2. Mutated `matmul_fwd_lowp` (llmm/matmul.mojo, the `comptime if not
         FP8_STATIC_SCALES:` block) to guard its `update_scale_pair[...]`
         call with `if input_state.step == 0:` -- so every call after the
         very first silently skips updating BOTH states: `step` freezes at
         1, the ring buffer never fills past its first slot, and `scale`
         freezes at call 0's own warmup value forever (the "wrapper skips
         update_scale when step > 0" bug this file's FIX 2 mutation
         targets). Rebuilt and reran: FAIL -- "AssertionError:
         input_state.step 1 != 16 after 16 warmup calls -- test setup
         assumption violated" (the setup-assumption assertion below, which
         caught the frozen-`step` bug immediately rather than surfacing as
         a confusing downstream scale mismatch). The four site tests above
         still PASSED in the same run (expected: each calls
         `matmul_fwd_lowp` only once per `AmaxState`, so `step == 0` on
         their only call and the mutation's `if` is never false for them)
         -- full-file summary "4 passed , 1 failed".
      3. llmm/matmul.mojo restored; `git diff --stat llmm/` empty
         afterward.
      4. Mutation (2) is a real catch, but it trips the setup-assumption
         guard rather than the substantive assertion -- it proves the test
         notices a state that stopped advancing, not that it notices a
         steady state computing the WRONG scale. So a second, sharper
         mutation was run against the branch itself: `_scale_from_history`
         (llmm/amax.mojo, `if step < history_len:`) forced to `if True:`,
         i.e. steady state behaves like warmup and uses this step's own
         amax. The `step` bookkeeping is untouched and the setup guard
         passes, so the discriminating assertion is what fires: FAIL --
         "steady-state input scale 4.172325e-07 != history-derived expected
         0.013671875 (relative error 0.9999695; this step's own spike amax
         was 1073741800.0)". 4.17e-07 is exactly `FMT_MAX * MARGIN / 2^30`,
         the spike-derived value. The four site tests passed in the same
         run. llmm/amax.mojo restored, diff empty.
    """
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime DT = DType.bfloat16
    comptime H = FP8_SPEC.amax_history_len  # 16
    comptime FMT_MAX = Float32(448.0)  # e4m3fn format max, FP8_SPEC.fwd_dtype
    comptime MARGIN_MULT = Float32(1 << FP8_SPEC.margin)  # margin=0 -> 1.0

    var rows = 32
    var in_channels = 64
    var out_channels = 64
    var n_in = rows * in_channels
    var n_w = out_channels * in_channels

    # Weight/bias: fixed, small, constant magnitude on EVERY call -- only
    # the input's magnitude varies across calls, so any drift in
    # `input_state.scale` can only be explained by `input_state`'s own
    # history, never by a coupling through `weight_state`.
    var host_w = ctx.enqueue_create_host_buffer[DT](n_w)
    for i in range(n_w):
        host_w.unsafe_ptr()[i] = Float32(0.015625).cast[DT]()  # 2^-6, exact
    var host_b = ctx.enqueue_create_host_buffer[DT](out_channels)
    for i in range(out_channels):
        host_b.unsafe_ptr()[i] = Float32(0.0).cast[DT]()

    var dev_w = ctx.enqueue_create_buffer[DT](n_w)
    var dev_b = ctx.enqueue_create_buffer[DT](out_channels)
    dev_w.enqueue_copy_from(host_w)
    dev_b.enqueue_copy_from(host_b)

    var dev_in = ctx.enqueue_create_buffer[DT](n_in)
    var dev_out = ctx.enqueue_create_buffer[DT](rows * out_channels)
    var dev_pre_gelu = ctx.enqueue_create_buffer[DT](rows * out_channels)
    var host_in = ctx.enqueue_create_host_buffer[DT](n_in)
    ctx.synchronize()

    var input_state = AmaxState[FP8_SPEC](ctx)
    var weight_state = AmaxState[FP8_SPEC](ctx)
    ctx.synchronize()

    # Warmup: H calls, constant-fill magnitude 2^0, 2^1, ..., 2^(H-1) -- each
    # an exact bf16 value (power of two), so `compute_amax` returns it
    # exactly and there is no rounding slop to account for below.
    for i in range(H):
        var v = Float32(1 << i)
        for j in range(n_in):
            host_in.unsafe_ptr()[j] = v.cast[DT]()
        dev_in.enqueue_copy_from(host_in)
        ctx.synchronize()
        matmul_fwd_lowp[DT, "gpu", use_gelu=False, has_bias=True](
            dev_out.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            input_state,
            weight_state,
            "test_steady_state",
            0,
            ctx,
        )
        ctx.synchronize()

    assert_true(
        input_state.step == H,
        "input_state.step "
        + String(input_state.step)
        + " != "
        + String(H)
        + " after "
        + String(H)
        + " warmup calls -- test setup assumption violated",
    )

    # 17th call: input's OWN amax this step is a spike (2^30), far above
    # everything in the history (max 2^(H-1) = 32768).
    var spike = Float32(1 << 30)
    for j in range(n_in):
        host_in.unsafe_ptr()[j] = spike.cast[DT]()
    dev_in.enqueue_copy_from(host_in)
    ctx.synchronize()
    matmul_fwd_lowp[DT, "gpu", use_gelu=False, has_bias=True](
        dev_out.unsafe_ptr().as_unsafe_any_origin(),
        dev_pre_gelu.unsafe_ptr().as_unsafe_any_origin(),
        dev_in.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        dev_w.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        dev_b.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        Int64(rows),
        Int64(1),
        Int64(in_channels),
        Int64(out_channels),
        input_state,
        weight_state,
        "test_steady_state",
        0,
        ctx,
    )
    ctx.synchronize()

    var scale = _read_f32(ctx, input_state.scale)

    var history_max = Float32(1 << (H - 1))  # 32768
    var expected_scale = FMT_MAX * MARGIN_MULT / history_max
    var warmup_style_scale = FMT_MAX * MARGIN_MULT / spike  # what a still-
    # warmup or frozen-scale bug would produce instead

    var rel_err = scale - expected_scale
    if rel_err < Float32(0.0):
        rel_err = -rel_err
    rel_err = rel_err / expected_scale
    assert_true(
        rel_err < Float32(0.01),
        "steady-state input scale "
        + String(scale)
        + " != history-derived expected "
        + String(expected_scale)
        + " (relative error "
        + String(rel_err)
        + "; this step's own spike amax was "
        + String(spike)
        + ")",
    )
    assert_true(
        scale > warmup_style_scale * Float32(1000.0),
        "steady-state input scale "
        + String(scale)
        + " is suspiciously close to the warmup/current-amax-based value "
        + String(warmup_style_scale)
        + " -- looks like this step's own spike leaked into its own scale"
        + " (delayed scaling defeated)",
    )


def main() raises:
    # fp8 GEMM is cuBLASLt-only (comptime assert HAS_CUBLAS in llmm/matmul.mojo);
    # comptime-gate discovery so non-CUDA compiles to a skip, not a build error.
    comptime if has_nvidia_gpu_accelerator():
        TestSuite.discover_tests[__functions_in_module()]().run()
    else:
        print(
            "SKIP tests/test_matmul_fwd_lowp.mojo: fp8/cuBLASLt is NVIDIA-only"
        )
