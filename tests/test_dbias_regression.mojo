# ===----------------------------------------------------------------------=== #
# tests/test_dbias_regression.mojo — regression cover for the dbias scratch
# overrun that silently zeroed bias gradients in production training runs
# (docs/ai/dbias_scratch_overrun_silent_zero_bug.md, fixed in e747faf).
#
# WHY A DEDICATED FILE. `matmul_bias_bwd` was already exercised by
# test_matmul_bwd_fp4.mojo and by the fp32 reference battery, and both were
# green for as long as the bug existed. Each missed it for a different reason,
# and this file is organized around not repeating either mistake:
#
#   * The fp4 test's `d_bias` check was RELATIVE — bf16 arm vs fp4 arm, both
#     running the same `matmul_bias_bwd`. A fault that zeroes both arms equally
#     passes by comparing 0 to 0. (That test now anchors to a host reduction
#     too; this file does not depend on it having done so.)
#   * The fp32/`B=4, T=64` reference batch was small in every dimension that
#     mattered: fp32 picks a host vector width that fits under the cap, and at
#     `T = 64` the overrunning row-blocks have no rows to sum, so their
#     out-of-bounds stores deposit `0.0` — which is the value the clobbered
#     counters were supposed to hold. The bug wrote out of bounds and was
#     harmless at exactly that shape.
#
# So every assertion here is anchored to a value computed on the HOST,
# independently of any GPU kernel, and every case runs at a production shape
# and precision rather than the cheapest one.
#
# WHAT IS ACTUALLY BEING GUARDED. Three distinct properties, in order of how
# badly their absence hurt:
#
#   1. `d_bias` is not trivially zero, and matches an independent host fp32
#      reduction of the same input. This is the assertion whose absence let a
#      dead kernel look correct for the life of the test.
#   2. Repeated use WITHIN ONE PROCESS stays correct. `DBIAS_COUNTERS` and
#      `DBIAS_SCRATCH` are persistent, process-lifetime buffers, so the failure
#      mode was "correct on first use, silently zero afterwards" — invisible to
#      any test that calls the kernel once. A training run is thousands of
#      steps in one process, which is why this property, not single-call
#      correctness, is the one that decides whether training is affected.
#   3. The widths that overran are covered. Scratch demand is
#      `FUSED_ROW_BLOCKS * out_channels`; only the wide biases (`3C` qkv, `4C`
#      fc) ever exceeded the old cap, and no test drove the kernel at those
#      widths with real rows behind them. Both GPT-2 scales this repo trains
#      are covered, at `rows = 4096` (B=4, T=1024), because the fault needs
#      overrunning row-blocks to have real rows to sum.
#
# VALIDATED AGAINST THE BUG, NOT JUST AGAINST THE FIX. A regression test that
# has only ever been run on fixed code is an assertion nobody has seen fail,
# which is the failure mode this whole file exists to answer. So it was run
# both ways. Reinstating the pre-fix state in `llmm/matmul.mojo` — the old
# hand-computed `DBIAS_SCRATCH_CAP = 1 << 20` with the capacity guard removed —
# and rebuilding gives:
#
#     PASS  test_dbias_124m_proj_width      (oc = C    = 768)
#     FAIL  test_dbias_124m_qkv_width       (oc = 3C   = 2304)   call 1/8
#     FAIL  test_dbias_124m_fc_width        (oc = 4C   = 3072)   call 1/8
#     PASS  test_dbias_774m_proj_width      (oc = C    = 1280)
#     FAIL  test_dbias_774m_qkv_width       (oc = 3C   = 3840)   call 1/8
#     FAIL  test_dbias_774m_fc_width        (oc = 4C   = 5120)   call 1/8
#
# Exactly the four widths that exceed the old cap fail, and the two that fit
# under it pass — the split the arithmetic predicts, from a test that knows
# nothing about the arithmetic. `llmm/matmul.mojo` was then restored and
# verified byte-identical to HEAD.
#
# One observation from that run is worth carrying: the pre-fix failures here
# surface as GARBAGE (`gpu[0] = 3.7e19` against a host value of `-22.6`), not
# as the all-zero signature seen in training. Same out-of-bounds write, but in
# this process it lands on something other than the arrival counters. That is
# the same "allocation layout decides the victim" behaviour that left the two
# fp8 production runs undamaged while bf16 and nvfp4 were hit, and it is why
# assertion (1) below cannot be the only check: a test that looked solely for
# zeros would have passed this run while the kernel returned 3.7e19.
#
# GPU-only; guarded by `has_nvidia_gpu_accelerator()` and expected to run under
# the shared-GPU flock like the rest of the GPU suites.
# ===----------------------------------------------------------------------=== #

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true

from llmm.matmul import matmul_bias_bwd


comptime DT = DType.bfloat16

# B=4, T=1024 — the real per-rank row count of a training step, and the
# smallest one at which the overrunning row-blocks have rows to sum. At
# `T = 64` (rows=256) this whole file would pass against the broken kernel.
comptime PROD_ROWS = 4096

# How many times each case re-runs the kernel in this one process. The counter
# buffer is allocated once per process, so a single call cannot distinguish
# "works" from "works once"; the original symptom was a kernel that was correct
# in isolation and dead on every subsequent use.
comptime REPEATS = 8


def _fill_random(
    ptr: Pointer[Scalar[DT], MutUntrackedOrigin],
    numel: Int,
    scale: Float32,
) -> None:
    for i in range(numel):
        var v = Float32((random_float64() * 2.0 - 1.0)) * scale
        ptr[unsafe_offset=i] = v.cast[DT]()


def _run_dbias_case(label: String, rows: Int, out_channels: Int) raises -> None:
    """Assert `matmul_bias_bwd` reproduces a host column-sum, on every call.

    The reference is a plain fp32 accumulation of the same `d_output` the GPU
    sees, done here on the host. Nothing in the reference path touches the code
    under test, so "GPU agrees with reference" cannot be satisfied by both sides
    being broken — which is precisely how the previous relative check failed.
    """
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()

    var n_out = rows * out_channels
    var host_doutput = ctx.enqueue_create_host_buffer[DT](n_out)
    ctx.synchronize()

    seed(20260727)
    _fill_random(host_doutput.unsafe_ptr(), n_out, Float32(0.5))

    var dev_doutput = ctx.enqueue_create_buffer[DT](n_out)
    dev_doutput.enqueue_copy_from(host_doutput)
    ctx.synchronize()

    # Host reference: sum each column over all rows, in fp32, from the same
    # bytes the GPU is about to read.
    var host_ref = ctx.enqueue_create_host_buffer[DType.float32](out_channels)
    ctx.synchronize()
    for c in range(out_channels):
        host_ref.unsafe_ptr()[unsafe_offset=c] = Float32(0.0)
    for r in range(rows):
        var row = host_doutput.unsafe_ptr().unsafe_offset(r * out_channels)
        for c in range(out_channels):
            host_ref.unsafe_ptr()[unsafe_offset=c] += row[unsafe_offset=c].cast[
                DType.float32
            ]()

    # Every column is a sum of `rows` random non-zero values; an exactly-zero
    # entry in the reference would make the "not trivially zero" assertion
    # below vacuous, so establish that it cannot happen before relying on it.
    for c in range(out_channels):
        assert_true(
            host_ref.unsafe_ptr()[unsafe_offset=c] != Float32(0.0),
            label
            + ": host reference column "
            + String(c)
            + " is exactly zero -- the non-zero assertion below would be"
            + " vacuous, so this input is unfit as a fixture",
        )

    var dev_d_bias = ctx.enqueue_create_buffer[DT](out_channels)
    var host_d_bias = ctx.enqueue_create_host_buffer[DT](out_channels)
    ctx.synchronize()

    for call in range(REPEATS):
        # Poison the destination first. Without this a kernel that never writes
        # would inherit whatever the previous iteration left behind and look
        # like it had produced a correct result.
        for c in range(out_channels):
            host_d_bias.unsafe_ptr()[unsafe_offset=c] = Float32(-12345.0).cast[
                DT
            ]()
        dev_d_bias.enqueue_copy_from(host_d_bias)
        ctx.synchronize()

        matmul_bias_bwd[DT, "gpu", accumulate=False](
            dev_d_bias.unsafe_ptr().as_unsafe_any_origin(),
            dev_doutput.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(out_channels),
            ctx,
        )
        ctx.synchronize()

        host_d_bias.enqueue_copy_from(dev_d_bias)
        ctx.synchronize()

        var suffix = " (call " + String(call + 1) + "/" + String(REPEATS) + ")"

        # (1) Not trivially zero. Stated separately from the numeric check
        # because "all zero" is the exact signature of the bug, and a reader
        # of a failure should see that named rather than inferred from an L2.
        var nonzero = 0
        for c in range(out_channels):
            if host_d_bias.unsafe_ptr()[unsafe_offset=c] != Scalar[DT](0):
                nonzero += 1
        assert_true(
            nonzero == out_channels,
            label
            + ": only "
            + String(nonzero)
            + " of "
            + String(out_channels)
            + " d_bias entries are non-zero"
            + suffix
            + " -- the finalize never ran (see"
            + " docs/ai/dbias_scratch_overrun_silent_zero_bug.md)",
        )

        # (2) Numerically right, against the host reduction. bf16 carries ~8
        # significant bits, and these are sums of 4096 terms, so compare in
        # relative L2 rather than exactly.
        var num = Float32(0.0)
        var den = Float32(0.0)
        for c in range(out_channels):
            var got = host_d_bias.unsafe_ptr()[unsafe_offset=c].cast[
                DType.float32
            ]()
            var want = host_ref.unsafe_ptr()[unsafe_offset=c]
            num += (got - want) * (got - want)
            den += want * want
        var rel_l2 = sqrt(num) / (sqrt(den) + Float32(1e-12))
        assert_true(
            rel_l2 < Float32(0.02),
            label
            + ": d_bias rel_l2 vs host reduction "
            + String(rel_l2)
            + " >= 0.02"
            + suffix
            + " (host[0]="
            + String(host_ref.unsafe_ptr()[unsafe_offset=0])
            + ", gpu[0]="
            + String(
                host_d_bias.unsafe_ptr()[unsafe_offset=0].cast[DType.float32]()
            )
            + ")",
        )

    _ = dev_doutput
    _ = dev_d_bias


# ===----------------------------------------------------------------------=== #
# GPT-2 124M (d12, C=768). `qkvb` and `fcb` are the two that overran the old
# cap; `attprojb`/`fcprojb` fit under it and died anyway, from sharing the
# counter array the wide calls had poisoned. Both classes are covered because
# the published checkpoints show both classes damaged.
# ===----------------------------------------------------------------------=== #


def test_dbias_124m_proj_width() raises:
    # attproj / fcproj bias: oc = C.
    _run_dbias_case("124M proj (oc=C=768)", PROD_ROWS, 768)


def test_dbias_124m_qkv_width() raises:
    # qkv bias: oc = 3C. Exceeded the old cap.
    _run_dbias_case("124M qkv (oc=3C=2304)", PROD_ROWS, 3 * 768)


def test_dbias_124m_fc_width() raises:
    # fc bias: oc = 4C, the widest in the model. Exceeded the old cap.
    _run_dbias_case("124M fc (oc=4C=3072)", PROD_ROWS, 4 * 768)


# ===----------------------------------------------------------------------=== #
# GPT-2 774M (d36, C=1280). The larger scale is not redundant: scratch demand
# is linear in `out_channels`, so 774M's 4C=5120 is the largest width this repo
# actually trains and the closest any test gets to the current cap.
# ===----------------------------------------------------------------------=== #


def test_dbias_774m_proj_width() raises:
    _run_dbias_case("774M proj (oc=C=1280)", PROD_ROWS, 1280)


def test_dbias_774m_qkv_width() raises:
    _run_dbias_case("774M qkv (oc=3C=3840)", PROD_ROWS, 3 * 1280)


def test_dbias_774m_fc_width() raises:
    _run_dbias_case("774M fc (oc=4C=5120)", PROD_ROWS, 4 * 1280)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
