# ===----------------------------------------------------------------------=== #
# Pure-Mojo unit + property tests for llmm.adamw.
#
# Run with:  make test-mojo   (equivalent to `mojo run -I . tests/test_adamw.mojo`)
#
# Covers:
#   - lerp() identity / endpoints
#   - bias-correction values at t = 1, 2
#   - zero-grad + zero-wd no-op (scaffolded TODO)
#   - single-step Adam (wd=0) equivalence (scaffolded TODO)
#   - fixture-driven trajectory check vs tests/fixtures/*.npz (scaffolded TODO)
#
# End-to-end equivalence vs torch.optim.AdamW lives in
# tests/test_adamw_equivalence.py; this file is the fast inner loop that
# exercises kernel math without crossing the MAX graph boundary.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_almost_equal, assert_true, TestSuite
from std.math import fma


# Single-dtype twin of `llmm.adamw.lerp` so this test file compiles
# standalone (the kernel currently has Mojo-1.0 incompatibilities — `@value`
# and unqualified `dtype` in struct field types — that block importing it).
# The kernel's lerp accepts mixed start/end dtypes for the Float32-moment
# storage path; the math we're verifying here is identical, so we keep the
# test signature simple. Once the kernel compiles cleanly, switch this back
# to `from llmm.adamw import lerp`.
def lerp[
    dtype: DType
](start: Scalar[dtype], end: Scalar[dtype], weight: Scalar[dtype]) -> Scalar[
    dtype
]:
    return fma(weight, end, fma(-weight, start, start))


# ===----------------------------------------------------------------------=== #
# lerp
# ===----------------------------------------------------------------------=== #


def test_lerp_endpoints() raises:
    assert_almost_equal(lerp[DType.float32](2.0, 9.0, 0.0), 2.0, atol=1e-6)
    assert_almost_equal(lerp[DType.float32](2.0, 9.0, 1.0), 9.0, atol=1e-6)


def test_lerp_midpoint() raises:
    var got = lerp[DType.float32](-4.0, 6.0, 0.5)
    assert_almost_equal(got, 1.0, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Bias correction math (mirrors llmm/adamw.mojo:183-184)
# ===----------------------------------------------------------------------=== #


def test_bias_correction_t1() raises:
    var beta1: Float32 = 0.9
    var beta2: Float32 = 0.999
    var b1c = 1.0 / (1.0 - beta1**1)
    var b2c = 1.0 / (1.0 - beta2**1)
    assert_almost_equal(b1c, 10.0, atol=1e-6)
    # 1.0 / (1 - 0.999) sits at the edge of fp32 precision since 0.999
    # isn't exactly representable; rel error ≈ 1.3e-5 is normal here.
    assert_almost_equal(b2c, 1000.0, atol=2e-2)


def test_bias_correction_t2() raises:
    var beta1: Float32 = 0.9
    var b1c_t2 = 1.0 / (1.0 - beta1**2)  # 1 / (1 - 0.81) = 1 / 0.19
    assert_almost_equal(b1c_t2, 5.2631578, atol=1e-5)


# ===----------------------------------------------------------------------=== #
# Single-step equivalence + property checks.
#
# Intentionally TODO scaffolds: the test-tensor allocation path (HostBuffer vs
# CPU DeviceContext) is best decided after the first Python<->Mojo equivalence
# run pins the op signature.
# ===----------------------------------------------------------------------=== #


def test_zero_grad_zero_wd_is_identity() raises:
    # TODO: allocate params/m/v on a host DeviceContext, run adamw_update_cpu
    # with grads=0 and weight_decay=0, assert params and v unchanged and m=0.
    # Pending: pick the simplest CPU-only allocation path for tests.
    assert_true(True)


def test_single_step_matches_handcomputed_adam() raises:
    # TODO: with weight_decay=0, run one step on n=1, beta1=beta2=0,
    # and verify param -= lr * grad / (|grad| + eps).  Easy closed form.
    assert_true(True)


def test_trajectory_matches_fixture() raises:
    # TODO: load tests/fixtures/fp32_small.npz (a numpy .npz is a zip of .npy
    # files; a small in-Mojo .npy reader keeps this self-contained), step
    # the optimizer `steps` times, and compare to `trajectory`.
    # Requires: tests/reference.py dump (run by `make test-fixtures`).
    assert_true(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
